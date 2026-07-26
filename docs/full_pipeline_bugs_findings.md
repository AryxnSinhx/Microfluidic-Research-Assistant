# Full-Pipeline Bug, Finding & Mitigation Log
### Microfluidics & Bilirubin-Detection RAFT Research Assistant

Covers every stage from chunk extraction through the 4-way ablation study
(Tier 1 -- the standalone QLoRA/Guanaco skill demonstration -- is intentionally
excluded, per scope). Stage 1 has no bugs on record; it's included as an
explicit row rather than silently omitted, flagged for input.

## Stage 1 -- Extracting Chunks from the 15 Research Papers

| Problem / Bug | Inference (Finding) | Mitigation | Result |
|---|---|---|---|
| No specific bugs recorded for this stage. | N/A -- not covered in this project's record. | N/A | Flagged for input -- add any PDF-parsing / chunk-boundary issues faced here. |

## Stage 2 -- Sanity Check

| Problem / Bug | Inference (Finding) | Mitigation | Result |
|---|---|---|---|
| Van den Bergh reaction answer hallucinated (wrong chemistry) during an early generation sanity check. | The base model's pretrained prior was bleeding through with no grounding in the actual source text. | Confirmed via a data audit that the training data itself was correct; identified this as an ungrounded-generation issue rather than a training-data error. | Directly motivated building the RAG pipeline in the first place. |
| Abstention probe failed -- the model confidently invented a fictional device rather than declining to answer an out-of-scope query. | The model was not reliably abstaining on out-of-scope queries on its own judgment alone. | Flagged for RAG plus an explicit, score-based abstention design rather than relying on the model to self-regulate. | Directly motivated the retrieval-score threshold safety net used in the final validated architecture. |

## Stage 3 -- Synthesizing RAFT Training Data (QA Generation)

| Problem / Bug | Inference (Finding) | Mitigation | Result |
|---|---|---|---|
| Abstention examples for all 15 withheld-source papers were written using one identical template ("...I'd need a source discussing [PAPER TITLE] to answer this..."). | Teaches the model a topic-to-abstention-template association rather than a general context-verification behavior. Not caught at synthesis time -- only identified retroactively by parsing the training data during the ablation stage. | Not yet remediated. Proposed fix: diversify abstention phrasing and add hard-negative examples where a paper's general topic is discussed but the specific requested fact is withheld, forcing context-checking rather than topic-matching. | Open item -- directly responsible for the false-abstention failure mode surfaced in Stage 10 (ablation study). |

## Stage 4 -- Compiling the Final Training Data Bucket / Mixture

| Problem / Bug | Inference (Finding) | Mitigation | Result |
|---|---|---|---|
| Bucket composition is heavily imbalanced (906 raft_domain / 453 sciq / 181 sciq_abstention / 136 camel_chemistry / 136 camel_physics = 1,812 total), and abstention-shaped examples are split across two buckets using two different templates. | sciq_abstention (181 examples) uses a generic "this specific topic" placeholder; raft_domain (97 examples) names a real corpus paper. Only the raft_domain-style template reproduces the specific false-abstention behavior seen at inference. | No changes made to bucket composition; confirmed via direct parsing of `final_training_mixture.jsonl` and cross-checked against expected counts. | Composition confirmed accurate; the two-template distinction directly informed the root-cause analysis. |

## Stage 5 -- Fine-Tuning Mistral on the Compiled Dataset (RAFT Training Run)

| Problem / Bug | Inference (Finding) | Mitigation | Result |
|---|---|---|---|
| T4 GPUs (Turing architecture) lack native BF16 tensor core support. | Training in bf16 would be numerically unsupported / inefficient on this hardware. | Used fp16 throughout training instead of bf16. | Training proceeded correctly on T4 hardware. |
| `device_map="auto"` across T4x2 produced naive sequential pipeline sharding with PCIe transfer overhead, not true data parallelism. | The multi-GPU setup was not actually accelerating training as intended. | Switched to single-GPU `device_map={"": 0}` for the training run. | More efficient, predictable single-GPU training. |
| Some training instances had their `[/INST]` response-template marker truncated beyond `MAX_SEQ_LENGTH`, causing `DataCollatorForCompletionOnlyLM` to silently mask the entire instance and produce NaN eval loss. | A subset of training examples were silently contributing zero loss signal, and eval-loss computation was being corrupted by the resulting NaNs. | Added pre-split filtering to catch and exclude/fix instances where the response template gets truncated beyond the sequence-length limit. | Eval loss computed correctly, with no silent zero-signal examples in the run. |
| Training loss decreased steadily from 0.4607 (step 20) to 0.1715 (step 100), while validation loss fluctuated without a clear downward trend: 0.4151 (step 20) -> 0.3713 (step 40) -> 0.4430 (step 60, an increase) -> 0.3572 (step 80) -> 0.3824 (step 100). | Steadily falling train loss alongside a fluctuating, non-monotonic validation loss (including an increase at step 60) is a classic divergence/overfitting signal, not steady joint improvement. | Used `load_best_model_at_end=True` so the checkpoint with the lowest recorded validation loss is selected rather than the final-epoch checkpoint. | Final merged model uses the best-validation checkpoint rather than a potentially overfit final epoch. |

## Stage 6 -- Model Push to Hub & Adapter Deployment

| Problem / Bug | Inference (Finding) | Mitigation | Result |
|---|---|---|---|
| `BadRequestError`: multipart upload part below S3 minimum size when pushing the merged model to the Hub. | A known bug in `huggingface_hub==0.24.6`'s multipart retry logic, not a problem with the model files themselves. | Switched to `api.upload_folder()` (streams files individually) instead of `push_to_hub()` / a zip-based approach. | Model uploaded successfully to the Hub. |
| `OSError`: no space left on device while zipping the merged model locally before upload. | Zipping duplicated the ~14GB model on disk, exceeding Kaggle's working-directory quota. | Abandoned the local-zip approach entirely; uploaded directly to the Hub via `upload_folder()`. | Eliminated the local-disk bottleneck entirely. |
| `Repository Not Found` error when pushing the LoRA adapter weights separately. | `upload_folder()` doesn't auto-create a repo the way `push_to_hub()` does. | Added an explicit `api.create_repo(..., exist_ok=True)` call before the upload. | Adapter weights pushed successfully to their own repo. |

## Stage 7 -- Inference Setup

| Problem / Bug | Inference (Finding) | Mitigation | Result |
|---|---|---|---|
| `ImportError`: bitsandbytes missing in a fresh notebook/environment. | A fresh environment without bitsandbytes installed, which prompted reconsidering whether 4-bit loading was actually necessary at all. | Initially considered a 4-bit load, but ultimately decided fp16-only was sufficient for single-prompt inference. | Simplified inference setup, and an early data point toward the eventual no-quantization decision. |
| CUDA OOM loading the merged model, with long RAG prompts (4 doc chunks). | Long prompts blew past the T4's free VRAM after model weights were already loaded. | Tried 4-bit -> 8-bit -> fp16 with `device_map="auto"` across 2xT4s. | Early precursor to both the final fp16/no-quantization architecture decision and the more thorough OOM mitigations later needed during the ablation study (Stage 10). |

## Stage 8 -- FAISS Chunking & Embeddings

| Problem / Bug | Inference (Finding) | Mitigation | Result |
|---|---|---|---|
| `ModuleNotFoundError: No module named 'faiss'` when running on Kaggle. | Kaggle's default environment doesn't ship faiss; not a code bug. | `pip install faiss-cpu` specifically (not faiss-gpu, since embedding/retrieval run on CPU by design). | Resolved; retrieval pipeline loads cleanly on every subsequent run. |
| `UnicodeDecodeError` ('utf-8' codec can't decode byte 0x80') when loading chunk metadata; also unclear which of two candidate files (`chunk_metadata.pkl` vs. `chunks.jsonl`) was authoritative. | A binary pickle file was being opened in text mode -- 0x80 is the pickle protocol marker, not corrupted data. Separately, `chunks.jsonl` turned out to be raw first-stage chunking output (pre-metadata), not usable as the retrieval source. | Rewrote the loader to open `chunk_metadata.pkl` in binary ('rb') mode; confirmed `chunks.jsonl` was not the intended metadata source. | Loader reliably reads the correct 903-chunk metadata with paper_title/section fields intact. |
| Confusion between the training QA data (`final_training_mixture.jsonl`) and the actual retrieval corpus -- the synthetic QA data was briefly mistaken for an indexable source. | The RAFT training data and the raw chunk corpus are easy to conflate since both derive from the same 15 papers, but only one is meant to be searched at inference time. | Clarified explicitly: only index the raw `chunk_metadata` file; never the QA training data. | Eliminated a category of retrieval bugs before they could occur. |

## Stage 9 -- Building the RAG Pipeline

| Problem / Bug | Inference (Finding) | Mitigation | Result |
|---|---|---|---|
| Retrieval for a PDMS-fabrication-steps question surfaces only Introduction / Table-of-Contents-level chunks, never the actual step-by-step methods section, even at TOP_K=5. | Not a retrieval-depth issue -- a retrieved Table-of-Contents chunk itself confirms the target section exists in the corpus, but its embedding doesn't rank near the top for this question's phrasing. Points to a chunking/embedding-quality issue specific to procedural/list-style text. | Deprioritized in favor of the higher-impact ablation investigation; documented as an open item. | Unresolved -- flagged for a future chunking-strategy revision (e.g., section-aware chunk boundaries). |
| False abstention despite correct retrieval -- prompt format mismatch: used `(paper_title)` instead of `(paper_title, section)`, which didn't match the RAFT training format exactly. | The fine-tuned model is sensitive to exact prompt structure; a mismatched format was indistinguishable, to the model, from missing context. | Added an explicit `section` field into the `[DOC N]` context blocks to match the training format exactly. | Eliminated this class of false abstention before the ablation study began. |
| Eponym mismatch: "Van den Bergh reaction" caused abstention even with the correct documents retrieved. | The corpus paper never uses the term "Van den Bergh reaction" (it uses "azobilirubin formation" instead); the model pattern-matches literally rather than semantically. | Confirmed as legitimate model behavior/limitation, not a bug -- documented as an ablation finding rather than something to fix. | Became a deliberately designed test case (E1/E2) in the ablation study. |
| Degenerate repetition loop -- the same sentence generated forever, a greedy-decoding failure mode on short abstention phrases. | Greedy decoding with no repetition control can get stuck repeating short phrases indefinitely. | Attempted `repetition_penalty` combined with `no_repeat_ngram_size`. | Stopped the infinite loop, but introduced a new problem (see next row). |
| Severe misspellings on technical terms ("bilirubin," "diazonium") across 4-bit, 8-bit, AND fp16 -- present regardless of quantization level. | `repetition_penalty=1.3` and `no_repeat_ngram_size=3` were both too aggressive, fighting the consistent subword token reuse that rare scientific vocabulary actually needs. | Lowered `repetition_penalty` to 1.1 and removed `no_repeat_ngram_size` entirely. | Correct spelling of rare scientific terms restored. |
| Repetition loop returned at the lower penalty; the base model (no fine-tune) produced even worse output (merged words, foreign characters) under the same settings. | Confirmed the fine-tune was helping fluency, not hurting it -- the repetition loop was a separate, generation-level issue from the spelling problem. | Added a post-hoc `truncate_repetition()` function to cut output at the first repeated sentence, instead of fighting the loop via generation parameters. | Clean output without sacrificing correct spelling -- used throughout the ablation study. |
| Fabricated non-existent paper title during abstention -- the model hallucinating a plausible-sounding but fake source name when uncertain. | Hallucinating source names when uncertain is a distinct risk from hallucinating chemistry -- a confident-sounding fake citation is arguably more dangerous than an honest "I don't know." | Added the retrieval-score threshold (0.4) -- skip generation entirely and return a fixed message if the top retrieval score is too low. | Eliminated fabricated citations in low-confidence cases; became a permanent safety net in the validated architecture. |

## Stage 10 -- Building the 4-Way Ablation Study

| Problem / Bug | Inference (Finding) | Mitigation | Result |
|---|---|---|---|
| CUDA OOM during ablation runs (14.51 / 14.56 GiB used on a T4). | Unquantized fp16 7B model weights alone nearly fill a 16GB T4; multi-chunk RAG context left no headroom for attention/KV-cache during generation. | Reduced TOP_K, capped per-chunk prompt length, set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`, switched to sdpa attention, added per-call tensor cleanup, and wrapped generation in try/except so one OOM doesn't kill the full run. | Full 17-question x 4-condition run (68 generations) completed with zero errors. |
| Synthesis test questions (S1/S2) still contained unfilled `[PAPER_A]`/`[PAPER_C]`-style placeholders on first run. | Placeholder text was never swapped for real paper titles before execution. | Replaced placeholders with real titles surfaced from the model's own retrieval hits in the prior run. | Synthesis questions became valid, real cross-paper tests. |
| `combined` model falsely claims retrieved documents don't contain an answer, even when they demonstrably do (S1, S2). | Root-caused by parsing the training set: all 97 raft_domain abstention examples share one template naming a real withheld paper, spanning all 15 corpus papers (ties back to Stage 3). | Built an isolation test set (S4/S5/S6) varying how many papers are explicitly named, to separate a phrasing-based trigger from a topic-memorization trigger. | Two distinct, partially-overlapping failure modes identified (see `ablation_writeup.md`). |
| Manual scoring initially rated 3 false-abstention responses (S1, S2, S6) as "appropriate," scoring them 4-5/5 with no hallucination flagged. | The scoring sheet surfaced the retrieval confidence score but not the actual list of retrieved paper titles. | Cross-referenced the scorer's notes against `retrieval_hits` from the raw run data and corrected the three rows' scores with evidence recorded in the notes field. | Headline ablation conclusion changed: combined no longer shows the best hallucination profile once scored accurately (41% vs. rag_only's 35%). |

## Bonus: Gradio Demo Deployment

| Problem / Bug | Inference (Finding) | Mitigation | Result |
|---|---|---|---|
| Private Hugging Face model repo caused a 401/repo-not-found error when loading the fine-tuned model in a fresh Kaggle session. | The merged model repo is private; an earlier session had been authenticated, but a fresh kernel doesn't retain that state. | Added Hugging Face Hub authentication via Kaggle Secrets (`kaggle_secrets.UserSecretsClient`) at the top of the script, executed before any model load. | Demo authenticates automatically at startup without exposing the token in the notebook. |

---
*See [`ablation study.md`]([ablation study.md](https://github.com/AryxnSinhx/Microfluidic-Research-Assistant/blob/7ddf5a7bd7e23db3fe4112cd00581608c40e33e4/docs/ablation%20study.md)) for the full ablation methodology and root-cause narrative.*
