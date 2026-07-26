# Microfluidics & Bilirubin-Detection RAFT Research Assistant

A domain-specialized research assistant combining Retrieval-Augmented Generation (RAG)
with Retrieval-Augmented Fine-Tuning (RAFT) over a 15-paper corpus spanning
microfluidic mixing, electrohydrodynamic instability, PDMS fabrication, and
bilirubin detection chemistry (the Van den Bergh / diazo reaction).

> **Scope note:** this repo covers Tier 2 (RAG) and Tier 3 (RAFT combined system)
> only. An earlier, isolated QLoRA fine-tune on the Guanaco conversational dataset
> (a standalone PEFT skill demonstration, unrelated to this domain) is intentionally
> excluded from this project.

## Architecture

```
                     ┌─────────────────────────┐
                     │   15 Source Papers (PDF) │
                     └────────────┬─────────────┘
                                  │ chunking
                                  ▼
                  ┌───────────────────────────────┐
                  │  903 chunks -> all-mpnet-base-v2 │
                  │  -> FAISS IndexFlatIP             │
                  └───────────────┬───────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                          │
        ▼                         ▼                          ▼
┌───────────────┐       ┌──────────────────┐      ┌────────────────────┐
│ RAG Pipeline   │       │ RAFT Fine-Tune    │      │ 4-Way Ablation      │
│ (retrieval +   │──────▶│ (Mistral-7B,      │─────▶│ Study                │
│  base model)   │       │  QLoRA r=16)      │      │ base/RAG/LoRA/       │
└───────────────┘       └──────────────────┘      │  combined            │
                                                    └──────────┬──────────┘
                                                               │
                                                               ▼
                                                    ┌────────────────────┐
                                                    │  Gradio Demo         │
                                                    │  (combined + cites)  │
                                                    └────────────────────┘
```

## Validated Architecture

| Component | Setting |
|---|---|
| Model | `aryxnsinhx/mistral-7b-microfluidics-raft` (merged), fp16, `device_map="auto"`, no quantization |
| Embeddings | `all-mpnet-base-v2`, CPU |
| Retrieval | FAISS `IndexFlatIP` over 903 raw corpus chunks |
| Prompt format | `[DOC N] (paper_title, section) text` -- must exactly match RAFT training format |
| Generation | Greedy (`do_sample=False`), `repetition_penalty=1.1`, no `no_repeat_ngram_size` |
| Safety nets | Retrieval-score threshold (0.4); post-hoc truncation at first repeated sentence |

## Headline Ablation Results

17 test questions across 4 categories (fluency, eponym-mismatch, abstention, synthesis),
evaluated across 4 conditions. Scores below are **corrected** -- an initial manual
scoring pass rated 3 false-abstention responses as "appropriate"; correcting those
against the raw retrieval evidence changed the conclusion (see
[`docs/ablation_writeup.md`]([docs/ablation_writeup.md](https://github.com/AryxnSinhx/Microfluidic-Research-Assistant/blob/ac3d80e94ea2ae8b93b4abe4adff5b009e53acdd/docs/ablation%20study.md)) for the full story).

| Condition | Avg. Fluency (1-5) | Avg. Grounding (1-5) | Hallucination Rate |
|---|---|---|---|
| base | 4.41 | 3.29 | 8/17 (47%) |
| rag_only | 4.59 | 4.18 | 6/17 (35%) |
| lora_only | 4.47 | 2.94 (lowest) | 10/17 (59%, highest) |
| combined (corrected) | 3.65 | 3.47 | 7/17 (41%) |

**Key finding:** the combined system falsely claims retrieved documents don't
contain an answer, in specific, identified conditions -- root-caused to a training-data
construction choice where every one of the 15 corpus papers was used as a
"withheld source" example with an identical abstention template. Full
investigation, isolation testing, and root-cause analysis in
[`docs/ablation_writeup.md`](docs/ablation_writeup.md).

## Repo Structure

- [`data pipeline/`](data pipeline/) -- chunk extraction, RAFT QA synthesis, final training mixture
- [`finetuning/`](finetuning/) -- Tier 3 RAFT fine-tune of Mistral-7B
- [`rag_pipeline/`](rag_pipeline/) -- retrieval + generation, the validated architecture
- [`ablation_study/`](ablation_study/) -- 4-way ablation harness, scoring tools, results
- [`demo/`](demo/) -- Gradio interface with source citations
- [`docs/`](docs/) -- full write-ups: ablation methodology/findings, and the complete pipeline bug/mitigation log

## Model & Data

- Fine-tuned model (merged): [`aryxnsinhx/mistral-7b-microfluidics-raft`](https://huggingface.co/aryxnsinhx/mistral-7b-microfluidics-raft) (private)
- Adapter weights: `aryxnsinhx/mistral-7b-microfluidics-raft-adapter` (private)
- FAISS index and training data are not committed to this repo -- see `data_pipeline/README.md` and `.gitignore` for details.
