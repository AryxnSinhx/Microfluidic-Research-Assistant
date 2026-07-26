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
[`docs/ablation_study.md`](docs/ablation_study.md) for the full story).

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
[`docs/ablation_study.md`](docs/ablation_study.md).

## Repo Structure

- [`data pipeline/`](<data pipeline/>) -- chunk extraction, RAFT QA synthesis, final training mixture (see [Data Pipeline](#data-pipeline) below)
- [`finetuning/`](finetuning/) -- Tier 3 RAFT fine-tune of Mistral-7B (see [Fine-Tuning](#fine-tuning-tier-3----raft-training-run) below)
- [`faiss indexing & rag/`](<faiss indexing & rag/>) -- retrieval + generation, the validated architecture
- [`ablation/`](ablation/) -- 4-way ablation harness, scoring tools, results (see [Ablation Study](#ablation-study) below)
- [`gradio demo/`](<gradio demo/>) -- Gradio interface with source citations
- [`graph & evaluation/`](<graph & evaluation/>) -- evaluation visualizations and supporting artifacts
- [`docs/`](docs/) -- full write-ups: [`ablation_study.md`](docs/ablation_study.md) (methodology + findings) and [`full_pipeline_bugs_findings.md`](docs/full_pipeline_bugs_findings.md) (complete pipeline bug/mitigation log)

## Model & Data

- Fine-tuned model (merged): [`aryxnsinhx/mistral-7b-microfluidics-raft`](https://huggingface.co/aryxnsinhx/mistral-7b-microfluidics-raft) (private)
- Adapter weights: `aryxnsinhx/mistral-7b-microfluidics-raft-adapter` (private)
- FAISS index and training data are not committed to this repo -- see the Data Pipeline section below and `.gitignore` for details.

---

## Data Pipeline

Chunk extraction from the 15 source papers, RAFT training-data synthesis
(QA generation), and compilation of the final training mixture. Lives in
[`data pipeline/`](<data pipeline/>).

### Provenance

- **Source:** 15 research papers spanning microfluidic mixing, electrohydrodynamic
  instability, PDMS fabrication, and bilirubin detection chemistry.
- **Chunking output:** 903 raw corpus chunks, each with `chunk_id`, `paper_title`,
  `section`, `text`, and `source_file` fields.
- **QA generation model:** Qwen2.5-14B-Instruct (local on Kaggle, 4-bit) --
  deliberately chosen to differ from the fine-tune target (Mistral-7B-v0.2)
  to avoid self-distillation contamination.

### Final Training Mixture Composition

`final_training_mixture.jsonl` -- 1,812 examples total:

| Bucket | Count | Notes |
|---|---|---|
| raft_domain | 906 | Domain-specific QA over the 15-paper corpus, including 97 abstention-labeled examples |
| sciq | 453 | General science QA |
| sciq_abstention | 181 | Generic abstention template ("...this specific topic...") |
| camel_chemistry | 136 | |
| camel_physics | 136 | |

### Known Issue: Uniform Abstention Template

**This is the single most important thing to know about this data before
reusing it.** All 97 `raft_domain` abstention-labeled examples share one
identical template:

> "The provided documents do not contain information to answer this question.
> I'd need a source discussing [PAPER TITLE] to answer this accurately."

The paper named in that slot spans **all 15 papers** in the corpus -- every
paper was used, at least twice, as a deliberately withheld source during
training. This is a reasonable way to teach abstention, but it has a
confirmed side effect: the fine-tuned model appears to associate a question's
*topic* with this memorized completion, rather than learning to check whether
the named paper is actually present in its current context. This was
root-caused during the ablation study (not caught at data-synthesis time) --
full investigation in [`docs/ablation_study.md`](docs/ablation_study.md).

**If regenerating this dataset:** the recommended fix is to diversify
abstention phrasing and add hard-negative examples where a paper's general
topic is discussed but the specific requested fact is withheld -- forcing the
model to learn context-verification rather than topic-matching.

### Retrieval vs. Training Data -- Do Not Conflate

The FAISS index used at inference time is built over the **903 raw corpus
chunks** (`chunk_metadata.pkl` in `faiss indexing & rag/`), never over
`final_training_mixture.jsonl`. These were briefly confused during pipeline
development -- see `docs/full_pipeline_bugs_findings.md`, Stage 8.

---

## Fine-Tuning (Tier 3 -- RAFT Training Run)

QLoRA fine-tune of `mistralai/Mistral-7B-v0.2` on the domain-specific RAFT
training mixture (see Data Pipeline above), producing the merged model used
throughout the RAG pipeline and ablation study. Lives in
[`finetuning/`](finetuning/).

> Not to be confused with an earlier, separate QLoRA fine-tune on the Guanaco
> conversational dataset -- that was an isolated PEFT skill demonstration and
> is out of scope for this repo.

### Config

See `finetuning/training_config.yaml`. Values are split into three groups in
that file: **Confirmed** (validated and documented -- base model, LoRA r=16,
fp16, single-GPU `device_map`, bucket composition, package versions),
**TODO** (real values used in the actual run that aren't on record --
`max_seq_length`, `num_train_epochs`, `save_steps`/`eval_steps`/`save_total_limit`,
LoRA alpha/dropout/target modules -- fill these in rather than trusting a guess),
and **Recommended future fix** (changes identified as worth making but not yet
applied to the run that produced the current checkpoints).

### Loss Curve

| Step | Training Loss | Validation Loss |
|---|---|---|
| 20 | 0.4607 | 0.4151 |
| 40 | 0.3661 | 0.3713 |
| 60 | 0.2553 | 0.4430 |
| 80 | 0.1706 | 0.3572 |
| 100 | 0.1715 | 0.3824 |
| 120 | 0.080000 | 0.424433 |
| 140 | 0.063000 | 0.442293 |

Training loss falls steadily; validation loss fluctuates without a clear
downward trend (notably increasing at step 60 before dropping again at step
80) -- a divergence/overfitting signal rather than steady joint improvement.
`load_best_model_at_end=True` was used to select the best-validation
checkpoint rather than the final-epoch one, though see the note in
`training_config.yaml` about a known `save_steps`/`eval_steps` misalignment
that may have limited how well this worked in practice.

### Key Hardware/Environment Notes

- **T4 lacks native BF16 tensor cores** (Turing architecture) -- fp16 must be
  used throughout training and inference; bf16 silently degrades performance.
- **`device_map="auto"` across T4x2 is the wrong choice for training** -- it
  produces naive sequential pipeline sharding with PCIe transfer overhead, not
  true data parallelism. Single-GPU `device_map={"": 0}` is correct here.
- **`bitsandbytes` must be left unpinned** on Kaggle's CUDA 12.8 environment --
  pinning to `0.43.1` causes a missing `libbitsandbytes_cuda128.so` error.

### Checkpoint Selection

The merged model pushed to HF Hub
(`aryxnsinhx/mistral-7b-microfluidics-raft`) uses the best-validation
checkpoint per `load_best_model_at_end=True`, not necessarily the final
training step -- see the loss table above for why that distinction matters
given the non-monotonic validation loss.

Full bug/mitigation log for this stage (S3 multipart upload error, disk-space
zip issue, repo-not-found on adapter push, the truncated-`[/INST]`/NaN-loss
collator bug, etc.) is in `docs/full_pipeline_bugs_findings.md`, Stages 5-7.

---

## Ablation Study

4-way ablation (base / rag_only / lora_only / combined) evaluating the
RAFT-fine-tuned model against a 17-question test bank across four categories:
core domain fluency, eponym-mismatch nuance, clean abstention probes, and
multi-paper synthesis. Lives in [`ablation/`](ablation/).

**Full methodology and findings write-up:** [`docs/ablation_study.md`](docs/ablation_study.md)

### Headline Finding

The `combined` condition (RAFT fine-tune + RAG) falsely claims retrieved
documents don't contain an answer, under specific, identified conditions --
root-caused to a training-data construction choice (every corpus paper used
as a "withheld source" example with an identical abstention template). See
the write-up for the full isolation-testing methodology and the
scoring-correction story.

### Files (in `ablation/`)

| File | Description |
|---|---|
| `ablation_study.py` | Main harness. Loads base and fine-tuned models sequentially (single T4, one 7B model in memory at a time), runs all 17 questions across all 4 conditions, handles retrieval, generation, and the retrieval-score abstention gate. |
| `build_scoring_sheet.py` | Converts raw `all_results.json` into a CSV template for manual scoring, with automated abstention-detection and grounding-token-overlap proxy metrics. |
| `check_training_data_composition.py` | Diagnostic script that parses the RAFT training mixture to check for comparison-phrased questions and extract the "withheld source" papers named in abstention-labeled examples -- this is what surfaced the root cause. |
| `ablation_scoring_tool.html` | Self-contained interactive scoring UI (no external dependencies). Loads all 68 rows (17 questions x 4 conditions) with pre-filled diagnostic context from the investigation, supports CSV export/import to resume a scoring session. |
| `results/all_results.json` | Raw output: every generation across all 4 conditions, including retrieval hits, scores, and timing. |
| `results/manual_scoring_sheet_corrected.csv` | Final scored sheet, with 3 rows (S1/S2/S6) corrected against retrieval evidence. |

### Running It

Requires the same Kaggle T4 environment as the fine-tuning stage, plus the
FAISS index and chunk metadata from `faiss indexing & rag/`.

```
pip install faiss-cpu -q
python ablation_study.py          # writes results/all_results.json
python build_scoring_sheet.py     # writes results/scoring_sheet.csv
```

Then either fill in `scoring_sheet.csv` by hand, or open
`ablation_scoring_tool.html` in a browser for the interactive version.

### Known Open Items

- The F3 test question (PDMS fabrication steps) exposes a retrieval-quality
  issue unrelated to the false-abstention finding -- see
  `docs/full_pipeline_bugs_findings.md`, Stage 9.
- The false-abstention pattern itself has not been remediated (see
  `docs/ablation_study.md`'s "Open Items" section for the proposed fix).
