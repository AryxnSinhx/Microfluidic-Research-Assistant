"""
STEP 3: Compile the final training dataset mixture.

Combines four sources into one shuffled SFT-ready JSONL:
  - Bucket A (~50%): your RAFT dataset (raft_dataset.jsonl from step 2) —
    domain-specific QA with retrieval distractors, teaches the model to
    ground answers in YOUR microfluidics/bilirubin papers specifically.
  - Bucket B (~25%): SciQ, reformatted into the same RAFT-style
    instruction/context/question/answer shape, with hard distractor
    passages — teaches general "ground your answer in provided evidence"
    behavior across a broad range of science topics.
  - Bucket C (~15%): CAMEL physics/chemistry, as plain SFT (no retrieval
    context) — teaches general scientific reasoning and problem-solving
    style, independent of retrieval.
  - Bucket D (~10%): distractor-only examples with no golden document —
    reinforces abstention (saying "I don't know" instead of hallucinating)
    using a different source than your RAFT abstention examples, for
    variety.

All four buckets get converted to ONE common schema before mixing, then
formatted into the final Mistral [INST]...[/INST] text strings your
QLoRA/SFTTrainer setup expects, then shuffled together.

Usage on Kaggle/Colab:
    !pip install datasets -q
"""
import json
import os
import random
import time
from collections import defaultdict

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from datasets import load_dataset

# ---------------------------------------------------------------------------
# CONFIG — edit these
# ---------------------------------------------------------------------------
RAFT_PATH = "/kaggle/input/your-dataset/raft_dataset.jsonl"
OUT_PATH = "/kaggle/working/final_training_mixture.jsonl"


USE_LOCAL_FILES = False                                #if Hugging Face Hub is flaky, set True after downloading SciQ/CAMEL elsewhere and uploading as Kaggle Dataset
LOCAL_SCIQ_PATH = "/kaggle/input/sciq-local/sciq"        
LOCAL_CAMEL_PATHS = {                                     
    "physics": "/kaggle/input/camel-local/physics",
    "chemistry": "/kaggle/input/camel-local/chemistry",
}

# Target proportions for the final mixture. These describe what fraction of
# the FINAL dataset each bucket should make up — the script back-solves how
# many examples to pull from SciQ/CAMEL based on how many RAFT examples you
# actually have, since bucket A's size is fixed (you already generated it).
TARGET_FRACTION_RAFT = 0.50     # bucket A
TARGET_FRACTION_SCIQ = 0.25     # bucket B
TARGET_FRACTION_CAMEL = 0.15    # bucket C
TARGET_FRACTION_ABSTENTION = 0.10  # bucket D (drawn from SciQ, distractor-only)

NUM_SCIQ_DISTRACTORS = 3       
CAMEL_SUBJECTS = ["physics", "chemistry"]  
RANDOM_SEED = 42

MAX_DOWNLOAD_RETRIES = 4
DOWNLOAD_RETRY_BACKOFF_SECONDS = 30 

random.seed(RANDOM_SEED)


def load_dataset_with_retry(path: str, split: str = "train", **kwargs):
    """
    Wraps load_dataset() with retries + backoff, since a flaky Kaggle<->Hub
    connection tends to stall or time out rather than fail fast — without
    this, one bad connection attempt kills the whole script after however
    long the download was crawling along before it gave up.
    """
    last_err = None
    for attempt in range(MAX_DOWNLOAD_RETRIES):
        try:
            return load_dataset(path, split=split, **kwargs)
        except Exception as e:
            last_err = e
            wait = DOWNLOAD_RETRY_BACKOFF_SECONDS * (2 ** attempt)
            print(f"  !! load_dataset({path!r}) failed (attempt {attempt+1}/{MAX_DOWNLOAD_RETRIES}): {e}")
            print(f"  -> waiting {wait}s before retry")
            time.sleep(wait)
    raise RuntimeError(
        f"Failed to load {path!r} after {MAX_DOWNLOAD_RETRIES} attempts. "
        f"Last error: {last_err}. If this keeps happening, Kaggle's connection to the "
        f"HF Hub may be degraded right now — consider setting USE_LOCAL_FILES = True "
        f"after downloading the dataset elsewhere and uploading it as a Kaggle Dataset."
    )

RAFT_INSTRUCTION = (
    "Answer the question using only the provided documents. "
    "Cite which document(s) support your answer. "
    "If the documents don't contain the answer, say so explicitly rather than guessing."
)
PLAIN_SFT_INSTRUCTION = "Answer the following question, showing your reasoning."
ABSTENTION_ANSWER_TEMPLATE = (
    "The provided documents do not contain information to answer this question. "
    "I'd need a source discussing {topic_hint} to answer this accurately."
)


# ---------------------------------------------------------------------------
# Common schema every bucket gets converted to before mixing:
#   {"instruction": str, "context": str | None, "question": str, "answer": str,
#    "metadata": {"bucket": str, "is_abstention": bool}}
# context is None for plain-SFT examples (bucket C) which have no retrieval
# documents at all — these get formatted differently at the very end.
# ---------------------------------------------------------------------------

def load_raft_bucket(path: str) -> list[dict]:
    """Bucket A: load your existing RAFT dataset as-is (already in the
    target shape from step 2)."""
    examples = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            ex["metadata"]["bucket"] = "raft_domain"
            examples.append(ex)
    return examples


def _load_sciq_source():
    """
    Load SciQ once, either from the Hub (with retry/backoff for flaky
    Kaggle<->Hub connectivity) or from local files if USE_LOCAL_FILES is
    set. Cached at module level so build_sciq_bucket and
    build_sciq_abstention_bucket don't each trigger their own separate
    download of the same dataset.
    """
    if USE_LOCAL_FILES:
        print(f"Loading SciQ from local path: {LOCAL_SCIQ_PATH}")
        sciq = load_dataset(LOCAL_SCIQ_PATH, split="train")
    else:
        print("Loading SciQ from the Hugging Face Hub...")
        sciq = load_dataset_with_retry("allenai/sciq", split="train")
    return sciq.filter(lambda r: r["support"] and len(r["support"].strip()) > 30)


def _load_camel_source(subject: str):
    """Load one CAMEL subject, either from the Hub (with retry) or local files."""
    if USE_LOCAL_FILES:
        path = LOCAL_CAMEL_PATHS[subject]
        print(f"Loading camel-ai/{subject} from local path: {path}")
        return load_dataset(path, split="train")
    print(f"Loading camel-ai/{subject} from the Hugging Face Hub...")
    return load_dataset_with_retry(f"camel-ai/{subject}", split="train")


_sciq_cache = None  


def _get_sciq():
    global _sciq_cache
    if _sciq_cache is None:
        _sciq_cache = _load_sciq_source()
    return _sciq_cache


def build_sciq_bucket(n_needed: int) -> list[dict]:
    """
    Bucket B: reformat SciQ into RAFT-style retrieval QA. SciQ's `support`
    field becomes the golden document; its three built-in distractor
    STRINGS aren't passages (they're wrong multiple-choice answers, not
    paragraphs), so for hard distractors we instead sample `support` text
    from OTHER SciQ rows — same trick as the RAFT distractor sampler, just
    applied to this dataset's own pool.
    """
    print(f"Building {n_needed} SciQ retrieval-QA examples...")
    sciq = _get_sciq()
    indices = list(range(len(sciq)))
    random.shuffle(indices)
    indices = indices[:n_needed]

    all_supports = sciq["support"]  # for distractor sampling
    examples = []
    for idx in indices:
        row = sciq[idx]
        golden_doc = f"[DOC 1] {row['support'].strip()}"
        distractor_idxs = random.sample(
            [i for i in range(len(all_supports)) if i != idx],
            min(NUM_SCIQ_DISTRACTORS, len(all_supports) - 1),
        )
        distractor_docs = [
            f"[DOC {i+2}] {all_supports[d].strip()}" for i, d in enumerate(distractor_idxs)
        ]
        doc_pool = [golden_doc] + distractor_docs
        random.shuffle(doc_pool)
        # re-label DOC numbers after shuffling so they're sequential
        relabeled = [f"[DOC {i+1}] {d.split(']', 1)[1].strip()}" for i, d in enumerate(doc_pool)]

        examples.append({
            "instruction": RAFT_INSTRUCTION,
            "context": "\n\n".join(relabeled),
            "question": row["question"].strip(),
            "answer": row["correct_answer"].strip().capitalize() + ".",
            "metadata": {"bucket": "sciq", "is_abstention": False},
        })
    return examples


def build_sciq_abstention_bucket(n_needed: int, exclude_questions: set[str]) -> list[dict]:
    """
    Bucket D: SciQ questions paired with ONLY distractor passages (no
    golden support document), teaching abstention from a different source
    than your RAFT abstention examples for variety in phrasing/style.
    """
    print(f"Building {n_needed} SciQ-based abstention examples...")
    sciq = _get_sciq()  # reuses the same load as build_sciq_bucket, no second download
    indices = [i for i in range(len(sciq)) if sciq[i]["question"] not in exclude_questions]
    random.shuffle(indices)
    indices = indices[:n_needed]

    all_supports = sciq["support"]
    examples = []
    for idx in indices:
        row = sciq[idx]
        # distractors only — deliberately exclude this row's own support
        distractor_idxs = random.sample(
            [i for i in range(len(all_supports)) if i != idx],
            min(NUM_SCIQ_DISTRACTORS + 1, len(all_supports) - 1),
        )
        doc_pool = [f"[DOC {i+1}] {all_supports[d].strip()}" for i, d in enumerate(distractor_idxs)]

        examples.append({
            "instruction": RAFT_INSTRUCTION,
            "context": "\n\n".join(doc_pool),
            "question": row["question"].strip(),
            "answer": ABSTENTION_ANSWER_TEMPLATE.format(topic_hint="this specific topic"),
            "metadata": {"bucket": "sciq_abstention", "is_abstention": True},
        })
    return examples


def build_camel_bucket(n_needed: int) -> list[dict]:
    """
    Bucket C: CAMEL physics/chemistry as plain SFT (no retrieval context at
    all) — teaches general scientific reasoning style, independent of
    retrieval. context is explicitly None here, distinct from "empty
    string", so downstream formatting can treat plain-SFT rows differently
    from retrieval rows with zero distractors.
    """
    per_subject = n_needed // len(CAMEL_SUBJECTS)
    examples = []
    for subject in CAMEL_SUBJECTS:
        ds = _load_camel_source(subject)
        indices = list(range(len(ds)))
        random.shuffle(indices)
        indices = indices[:per_subject]
        for idx in indices:
            row = ds[idx]
            examples.append({
                "instruction": PLAIN_SFT_INSTRUCTION,
                "context": None,
                "question": row["message_1"].strip(),
                "answer": row["message_2"].strip(),
                "metadata": {"bucket": f"camel_{subject}", "is_abstention": False},
            })
    return examples



# Final text formatting (Mistral [INST]...[/INST] format)

def format_example(ex: dict) -> str:
    """
    Convert one common-schema example into the final training string.
    Retrieval examples (context is not None) get the instruction + context
    + question all inside [INST]; plain-SFT examples (context is None,
    i.e. CAMEL) skip the context block entirely rather than printing an
    empty "Context:" section, since that would teach the model the
    retrieval-prompt format is sometimes contentless, which isn't a
    pattern you want it to learn.
    """
    if ex["context"] is not None:
        prompt = f"{ex['instruction']}\n\n{ex['context']}\n\nQuestion: {ex['question']}"
    else:
        prompt = f"{ex['instruction']}\n\nQuestion: {ex['question']}"
    return f"<s>[INST] {prompt} [/INST] {ex['answer']}</s>"



# Main

def main():
    raft_examples = load_raft_bucket(RAFT_PATH)
    n_raft = len(raft_examples)
    print(f"Loaded {n_raft} RAFT examples (bucket A, fixed size)")

    # Back-solve target total dataset size from bucket A's fixed size and
    # its target fraction, then derive how many examples the other buckets
    # need to hit THEIR target fractions of that same total.
    if TARGET_FRACTION_RAFT <= 0:
        raise ValueError("TARGET_FRACTION_RAFT must be > 0 to size the rest of the mixture")
    target_total = round(n_raft / TARGET_FRACTION_RAFT)
    n_sciq = round(target_total * TARGET_FRACTION_SCIQ)
    n_camel = round(target_total * TARGET_FRACTION_CAMEL)
    n_abstain = round(target_total * TARGET_FRACTION_ABSTENTION)

    print(f"Target total: ~{target_total} examples")
    print(f"  bucket A (RAFT, domain):     {n_raft}")
    print(f"  bucket B (SciQ, retrieval):  {n_sciq}")
    print(f"  bucket C (CAMEL, plain SFT): {n_camel}")
    print(f"  bucket D (SciQ, abstention): {n_abstain}")

    sciq_examples = build_sciq_bucket(n_sciq)
    sciq_questions_used = {ex["question"] for ex in sciq_examples}
    abstain_examples = build_sciq_abstention_bucket(n_abstain, exclude_questions=sciq_questions_used)
    camel_examples = build_camel_bucket(n_camel)

    all_examples = raft_examples + sciq_examples + abstain_examples + camel_examples
    random.shuffle(all_examples)

    print(f"\nFinal mixture: {len(all_examples)} examples")
    bucket_counts = defaultdict(int)
    for ex in all_examples:
        bucket_counts[ex["metadata"]["bucket"]] += 1
    for bucket, count in sorted(bucket_counts.items()):
        pct = 100 * count / len(all_examples)
        print(f"  {bucket:20s} {count:6d}  ({pct:.1f}%)")

    with open(OUT_PATH, "w") as f:
        for ex in all_examples:
            f.write(json.dumps({"text": format_example(ex), "metadata": ex["metadata"]}) + "\n")

    print(f"\nWrote {len(all_examples)} formatted training examples to {OUT_PATH}")
    print("\nSample formatted example:")
    print(format_example(all_examples[0])[:600])


if __name__ == "__main__":
    main()