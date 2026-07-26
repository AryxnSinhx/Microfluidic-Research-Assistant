import json
from collections import Counter, defaultdict

chunks = [json.loads(l) for l in open(r"data pipeline/1. extraction & chunking/result/chunks.jsonl")]
print(f"Total chunks: {len(chunks)} from {len(set(c['paper_title'] for c in chunks))} papers")

# 1. Which papers fell back to no-heading-detected?
fallback_papers = {c['paper_title'] for c in chunks if "no headings detected" in c['section']}
print("\nFell back to flat body (need manual check):")
for p in fallback_papers:
    print(" -", p)

# 2. Per-paper chunk count and section list — eyeball for anything looking off
by_paper = defaultdict(list)
for c in chunks:
    by_paper[c['paper_title']].append(c)

for paper, paper_chunks in by_paper.items():
    sections = [c['section'] for c in paper_chunks]
    print(f"\n{paper}: {len(paper_chunks)} chunks, {len(set(sections))} distinct sections")
    for s in sorted(set(sections)):
        print("   -", s[:90])

# 3. Flag suspiciously tiny or suspiciously huge chunks (extraction artifacts)
sizes = [len(c['text']) for c in chunks]
print(f"\nChunk size range: {min(sizes)}–{max(sizes)} chars")
weird = [c for c in chunks if len(c['text']) < 80 or len(c['text']) > 2500]
if weird:
    print(f"\n{len(weird)} chunks outside normal size range — check these:")
    for c in weird[:10]:
        print(f"  [{c['paper_title']} / {c['section']}] ({len(c['text'])} chars): {c['text'][:100]}")
