# Microfluidics & Bilirubin-Detection RAFT Research Assistant
### Retrieval-Augmented Fine-Tuning (RAFT) System -- Ablation Study, Root-Cause Analysis, and Findings

*Portfolio write-up. Tier 1 (a standalone QLoRA fine-tune on the Guanaco
conversational dataset, an isolated PEFT skill demonstration) is intentionally
excluded from this write-up.*

## 1. Project Overview

This project builds a domain-specialized research assistant over a 15-paper
corpus spanning microfluidic mixing, electrohydrodynamic instability, PDMS
fabrication, and bilirubin detection chemistry (the Van den Bergh / diazo
reaction). It combines two components: a from-scratch retrieval-augmented
generation (RAG) pipeline built without high-level abstractions (no
LangChain/LlamaIndex), and a RAFT-style (Retrieval-Augmented Fine-Tuning)
domain fine-tune of Mistral-7B that learns to reason jointly over retrieved
context and to abstain when context is insufficient.

The system was evaluated with a 4-way ablation study -- base model, RAG-only,
fine-tune-only (LoRA-only), and the combined RAFT+RAG system -- across 17 test
questions spanning four categories: core domain fluency, an eponym-mismatch
nuance (chemistry present in the corpus under different terminology), clean
out-of-scope abstention probes, and multi-paper synthesis questions.

### Validated Architecture

- **Model:** `aryxnsinhx/mistral-7b-microfluidics-raft` (merged), fp16, `device_map="auto"`, no quantization
- **Embeddings:** `all-mpnet-base-v2`, CPU
- **Retrieval:** FAISS `IndexFlatIP` over 903 raw corpus chunks
- **Prompt format:** `[DOC N] (paper_title, section) text` -- must exactly match RAFT training format
- **Generation:** greedy decoding, `repetition_penalty=1.1`, no ngram blocking (both preserve correct spelling of rare terms like "bilirubin" and "diazonium")
- **Safety nets:** retrieval-score threshold (0.4) to block generation on low-confidence retrieval; post-hoc truncation at the first repeated sentence

## 2. Ablation Results

Aggregate manual-scoring results across all four conditions, **after correcting**
a scoring blind spot described in Section 4:

| Condition | Avg. Fluency (1-5) | Avg. Grounding (1-5) | Hallucination Rate |
|---|---|---|---|
| base | 4.41 | 3.29 | 8/17 (47%) Yes+Partial |
| rag_only | 4.59 | 4.18 | 6/17 (35%) Yes+Partial |
| lora_only | 4.47 | 2.94 (lowest) | 10/17 (59%, highest) Yes+Partial |
| combined (corrected) | 3.65 | 3.47 | 7/17 (41%) Yes+Partial |

*Hallucination rate combines 'Yes' and 'Partial' ratings out of 17 questions per condition.*

The pattern is consistent with expectations for the components that are
working correctly: retrieval grounding (present in `rag_only` and `combined`)
is what drives the grounding-score improvement over base; fine-tuning without
retrieval (`lora_only`) is the weakest condition on both grounding and
hallucination, since it has nothing to check its own recall against. What does
**not** hold up under correction is the initial impression that the combined
system was straightforwardly the best-calibrated condition -- see Section 4.

## 3. Ablation Question Design

17 questions across four categories, designed to isolate specific behaviors
rather than only measure generic answer quality:

- **Fluency (5 questions):** core domain concepts with a single clear grounded answer, testing baseline correctness and fluency.
- **Eponym-mismatch (3 questions):** the same underlying chemistry asked once via a name absent from the source text ("Van den Bergh reaction") and once via terminology present in the corpus ("diazo reaction") -- the model should abstain on the former and answer the latter.
- **Abstention probes (3 questions):** genuinely out-of-scope or fictional queries, testing clean refusal without fabrication.
- **Synthesis (6 questions):** multi-paper questions requiring information from two or more sources, including a targeted isolation set (see Section 4) varying how many source papers are explicitly named in the question.

## 4. Key Finding: False-Abstention Root Cause

The combined system was observed to falsely claim that retrieved documents did
not contain an answer, in cases where the needed papers were demonstrably
present in its context -- confirmed by inspecting the raw `retrieval_hits` for
each generation. In one case (S2), both papers named in the question were
literally the top-2 highest-scoring retrieved chunks, and `rag_only`, given the
identical retrieved context, answered the question correctly. This ruled out
retrieval failure or prompt-format mismatch as explanations.

Parsing the 1,812-example RAFT training set traced the behavior to a specific
construction choice: all 97 abstention-labeled examples in the `raft_domain`
bucket share one template ("...I'd need a source discussing [PAPER TITLE] to
answer this..."), and the paper named in that slot spans **all 15 papers** in
the corpus -- every paper had been used, at least twice, as a deliberately
withheld source during training. This is a reasonable way to teach abstention,
with a plausible side effect: the model may have learned to associate a
question's topic with a memorized completion naming a specific paper, rather
than learning to check whether that paper is actually present in the current
context.

A targeted isolation test separated two distinct, partially overlapping
failure modes:

| ID | Question shape | Papers named | Outcome (combined) |
|---|---|---|---|
| S1 | Two papers named, mixing <-> bilirubin sensitivity | 2 | False abstention (partial basis -- one of two papers genuinely not retrieved) |
| S2 | Two papers named, PDMS <-> electrohydrodynamic instability | 2 | False abstention (both papers were top-2 retrieved chunks) |
| S4 | Same intent as S1, zero papers named | 0 | Correctly grounded |
| S5 | One paper named, single-source (not comparison) question | 1 | Correctly grounded |
| S6 | Same intent as S2, zero papers named | 0 | False abstention anyway -- named a third, unmentioned paper from context |

*S4 and S6 hold the same underlying synthesis intent as S1 and S2 respectively,
with zero papers named in the question, to isolate whether explicit naming is
the trigger.*

The results support two coexisting mechanisms rather than one simple rule:
(1) explicitly naming two paper titles in a comparison question is a reliable,
mechanical trigger (S1, S2 both fail; S5, with exactly one paper named, does
not -- ruling out "any naming" as the explanation); and (2) independent of
naming, certain topic pairings tied to heavily-represented training examples
can trigger the same behavior (S6 fails despite naming nothing), though this
is inconsistent rather than universal (S4 does not fail). A follow-up check for
a specific phrasing pattern ("compatible"/"intersect" framing) found no support
in the training data, so that hypothesis was explicitly ruled out rather than
assumed.

### Scoring Correction

The initial manual scoring pass rated the S1, S2, and S6 false-abstention
responses as "appropriate" (4-5/5, no hallucination flagged), because the
scoring sheet surfaced the retrieval confidence score but not the actual list
of retrieved paper titles -- a scorer without that cross-reference has no way
to see that the cited "missing" source was, in fact, present. Correcting these
three rows against the raw `retrieval_hits` data changed the headline result:
`combined` no longer shows the best hallucination profile of the four
conditions (see Section 2). This is treated as a stronger and more defensible
result than the uncorrected version, since it reflects an actual failure mode
rather than an artifact of incomplete scoring context.

## 5. Conclusions

The retrieval-augmented components of the system work as intended: RAG
grounding measurably improves factual accuracy over the base model, and the
combined RAFT+RAG system achieves comparable grounding while introducing
calibration behavior (abstention) that base and RAG-only do not have. However,
that calibration is not yet reliable -- it both correctly declines genuinely
out-of-scope and eponym-mismatched questions, and incorrectly declines
legitimate, well-supported synthesis questions under specific, identified
conditions.

The value of this ablation was less in confirming that fine-tuning "worked"
and more in surfacing, isolating, and root-causing a specific failure mode
back to a concrete training-data construction choice -- and in catching and
correcting a scoring blind spot that would otherwise have overstated the
system's reliability.

### Open Items

- Remediation of the false-abstention pattern (e.g., hard-negative training examples where a paper's topic is discussed but the specific fact is withheld, forcing context-verification rather than topic-matching) has not yet been attempted.
- The F3 chunking/embedding-quality issue (fabrication-methods content not surfacing in retrieval even at TOP_K=5) remains unresolved.
- A Gradio demo with source citations has been built on the same validated pipeline for interview/portfolio use.

---
*See [`full_pipeline_bugs_findings.md`](full_pipeline_bugs_findings.md) for the complete stage-by-stage bug/mitigation log across the entire pipeline.*
