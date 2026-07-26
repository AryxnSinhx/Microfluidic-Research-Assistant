import json
import re
from collections import Counter, defaultdict
from pathlib import Path

TRAINING_DATA_PATH = r"C:\Users\sinha\Desktop\Research Assistant\content\final_training_mixture.jsonl"

Q_RE = re.compile(r"Question:\s*(.*?)\s*\[/INST\]", re.DOTALL | re.IGNORECASE)
A_RE = re.compile(r"\[/INST\]\s*(.*?)(?:</s>|$)", re.DOTALL)
NAMED_SOURCE_RE = re.compile(r"I'd need a source discussing (.+?)(?:\s+to answer|\.)", re.IGNORECASE)

COMPARISON_PATTERNS = [
    r"\bcompare\b", r"\bcomparison\b", r"\bcontrast\b", r"\bversus\b", r"\bvs\.?\b",
    r"\bboth (documents|papers|sources|studies)\b",
]
COMPARISON_RE = re.compile("|".join(COMPARISON_PATTERNS), re.IGNORECASE)

QUOTED_TITLE_RE = re.compile(r"['\"]([^'\"]{25,150})['\"]")


def load_records(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- update TRAINING_DATA_PATH.")
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    records = load_records(TRAINING_DATA_PATH)
    print(f"Loaded {len(records)} training records.\n")

    bucket_counts = Counter()
    comparison_by_bucket = defaultdict(int)
    abstention_by_bucket = defaultdict(int)
    quoted_title_by_bucket = defaultdict(int)
    named_paper_counts = Counter()
    parse_fail = 0

    for r in records:
        text = r.get("text", "")
        meta = r.get("metadata", {})
        bucket = meta.get("bucket", "UNKNOWN")
        is_abstention_flag = bool(meta.get("is_abstention", False))
        bucket_counts[bucket] += 1

        qm = Q_RE.search(text)
        am = A_RE.search(text)
        if not qm or not am:
            parse_fail += 1
            continue
        question, answer = qm.group(1), am.group(1)

        if COMPARISON_RE.search(question):
            comparison_by_bucket[bucket] += 1
        if QUOTED_TITLE_RE.search(question):
            quoted_title_by_bucket[bucket] += 1
        if is_abstention_flag:
            abstention_by_bucket[bucket] += 1
            if bucket == "raft_domain":  
                m = NAMED_SOURCE_RE.search(answer)
                if m:
                    named_paper_counts[m.group(1).strip()] += 1

    print(f"Parse failures: {parse_fail}\n")

    print("=== Bucket distribution ===")
    for b, c in bucket_counts.most_common():
        print(f"  {b:20s}: {c}")

    print("\n=== 'Compare X vs Y' phrased questions, by bucket ===")
    print("(tests whether comparison PHRASING itself is rare -- it isn't, see below)")
    for b, c in bucket_counts.most_common():
        cc = comparison_by_bucket.get(b, 0)
        print(f"  {b:20s}: {cc:4d} / {c:4d}  ({cc/c*100:.1f}%)")

    print("\n=== Questions quoting a long string (25-150 chars, i.e. plausibly a title) ===")
    print("(tests whether literal title-quoting is rare -- it is, near-zero)")
    for b, c in bucket_counts.most_common():
        qc = quoted_title_by_bucket.get(b, 0)
        print(f"  {b:20s}: {qc:4d} / {c:4d}")

    print("\n=== Abstention-labeled examples, by bucket ===")
    for b, c in bucket_counts.most_common():
        ac = abstention_by_bucket.get(b, 0)
        print(f"  {b:20s}: {ac:4d} / {c:4d}")
    print("  NOTE: two distinct abstention templates exist. sciq_abstention (181)")
    print("  uses a generic placeholder: '...I'd need a source discussing this")
    print("  specific topic to answer this accurately.' raft_domain (97) names a")
    print("  REAL corpus paper instead. Both share the identical opening sentence")
    print("  ('The provided documents do not contain information...'), which is")
    print("  15.3% of the entire training set (278/1812) -- likely why that exact")
    print("  phrase is so reliably produced whenever the model decides to abstain.")

    print("\n=== KEY FINDING: unique 'missing source' papers named in raft_domain")
    print("    abstention examples (template: \"I'd need a source discussing X\") ===")
    for paper, c in named_paper_counts.most_common():
        print(f"  {c:3d}x  {paper}")
    print(f"\n  Total unique papers named as a withheld/missing source: {len(named_paper_counts)}")
    print("  If this covers most/all of your 15-paper corpus, the model likely learned")
    print("  a per-paper topic -> abstention-template association during training that")
    print("  doesn't check whether that paper is ACTUALLY in the current context --")
    print("  which would explain false abstention on F3/S1/S2 regardless of whether")
    print("  the question names a paper explicitly or not. Cross-check against the")
    print("  S4/S5/S6 isolation results (paper-agnostic phrasing, same underlying")
    print("  papers) to confirm: if S4/S6 STILL false-abstain despite naming nothing,")
    print("  that rules out 'named-source phrasing' as the trigger and confirms this")
    print("  topic-keyed memorization theory instead.")


if __name__ == "__main__":
    main()