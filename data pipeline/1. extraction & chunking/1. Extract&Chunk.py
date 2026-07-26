import json
import os
import re
import statistics
import uuid
from collections import Counter
from pathlib import Path

import pdfplumber
import wordninja
from spellchecker import SpellChecker

_SPELL = SpellChecker()

# ---------------------------------------------------------------------------
# CONFIG — edit these
# ---------------------------------------------------------------------------
PDF_DIR = r"directory\papers"          
OUT_PATH = r"your_desired_result directory\chunks.jsonl"   
CHUNK_TOKENS = 350                   
OVERLAP_TOKENS = 50                  
MIN_CHUNK_CHARS = 60                 
HEADING_SIZE_RATIO = 1.08             
MAX_HEADING_WORDS = 12                


EXCLUDED_SECTION_KEYWORDS = re.compile(
    r"\b(References|Bibliography|Acknowledg[e]?ments?)\b", re.IGNORECASE
)

CHARS_PER_TOKEN = 4
CHUNK_CHARS = CHUNK_TOKENS * CHARS_PER_TOKEN
OVERLAP_CHARS = OVERLAP_TOKENS * CHARS_PER_TOKEN

KNOWN_HEADING_WORDS = re.compile(
    r"\b(Abstract|Introduction|Background|Related Work|Methods?|"
    r"Materials and Methods|Methodology|Experimental(?: Setup)?|"
    r"Results?(?: and Discussion)?|Discussion|Conclusion[s]?|"
    r"Acknowledg[e]?ments?|References|Appendix)\b",
    re.IGNORECASE,
)

NUMBERED_HEADING_PREFIX = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s*(?=[A-Z])")  


_LIGATURE_CANDIDATES = ["ffi", "ffl", "ff", "fi", "fl", "st", "ct", "tt"]
_CID_GAP_PATTERN = re.compile(r"\S*\(cid:\d+\)\S*")


CONCAT_WORD_MIN_LEN = 25


def repair_concatenated_words(text: str) -> str:
    """
    Safety net for words that pdfplumber's extraction merged together with
    no space — this can happen on dense/small-font academic PDFs even with
    x_tolerance_ratio tuned at extraction time (no single tolerance value
    is robust against arbitrarily tight kerning). Only touches alphabetic
    runs at least CONCAT_WORD_MIN_LEN characters long, so real long
    scientific words are left alone; only word runs unambiguously too long
    to be one real word get split.
    """
    def repair_token(m: re.Match) -> str:
        token = m.group(0)
        split = wordninja.split(token)
        if len(split) <= 1:
            return token  # wordninja found no good split — leave it as-is
        return " ".join(split)

    return re.sub(rf"[a-zA-Z]{{{CONCAT_WORD_MIN_LEN},}}", repair_token, text)


PAGE_HEADER_PREFIX = re.compile(
    r"^\d{1,4}\s+[A-Z]{2,}(\s*[¥•·\u00b7\u2022]\s*[A-Z]{2,})+\s*[—\-]?\s*"
)


def strip_page_header_prefix(text: str) -> str:
    """Remove a leading page-number+author-names running-header watermark
    from a line, if present, returning whatever real content (if any)
    follows it on the same extracted line."""
    return PAGE_HEADER_PREFIX.sub("", text, count=1)


def fix_cid_ligatures(text: str) -> str:
    """
    Repair "(cid:NNN)" gaps left when a PDF's embedded font subset can't be
    mapped to Unicode for ligature glyphs (fi, fl, ffi, etc.) — common in
    academic PDFs with custom/subset fonts. The CID number itself is
    font-specific and not portable across different PDFs, so we don't trust
    it directly. Instead we try each common ligature in the gap and keep
    whichever produces a real dictionary word (using an offline spellchecker
    dictionary, no network calls). Falls back to just removing the marker if
    no candidate matches a real word — better than leaving "(cid:222)" in
    text that gets fed to an LLM for QA generation.
    """
    def repair_match(m: re.Match) -> str:
        word_with_gap = m.group(0)
        parts = re.split(r"\(cid:\d+\)", word_with_gap)
        if len(parts) != 2:
            return re.sub(r"\(cid:\d+\)", "", word_with_gap)  
        before, after = parts
        for lig in _LIGATURE_CANDIDATES:
            candidate = before + lig + after
            check_word = re.sub(r"[^a-zA-Z]", "", candidate)
            if check_word and check_word.lower() in _SPELL:
                return candidate
        return before + after 

    return _CID_GAP_PATTERN.sub(repair_match, text)


def looks_like_byline_or_junk(text: str) -> bool:
    """
    Detect title-page noise (author bylines, affiliations, footnote markers,
    page numbers/DOIs) that can otherwise be mistaken for a heading because
    it shares the same large/bold font on a title page. Real section
    headings ("Closed-Channel Microreactors", "2.1 Capillary-Based...")
    don't carry these fingerprints, so this check runs alongside — not
    instead of — the typography check.
    """
    stripped = text.strip()
    if not stripped:
        return True

    if re.match(r"^\s*Author Manuscript\b", stripped, re.IGNORECASE):
        return True

    # Footnote/affiliation markers stuck to a name: "Whitesides 1 *", "Smith2,3"
    has_trailing_marker = bool(re.search(r"[A-Za-z]\s*[\d*]{1,3}\s*[*,]?\s*$", stripped))
    has_initial_name_pattern = bool(re.search(r"\b[A-Z][a-z]+\s+[A-Z]\.\s*[A-Z][a-z]+\b", stripped))
    if has_trailing_marker and has_initial_name_pattern:
        return True
    if has_initial_name_pattern and len(stripped.split()) <= 10:
        return True

    # Mostly non-alphabetic content (page numbers, DOIs, bare dates)
    alpha_chars = sum(c.isalpha() for c in stripped)
    if alpha_chars / len(stripped) < 0.5:
        return True

    return False


def looks_like_real_heading_text(text: str) -> bool:
    """
    Require a heading candidate to contain actual word-like content, not
    just be short and not end in a period. Inline math/equation fragments
    are routinely rendered in a different font or size than body prose
    (italic math fonts, symbol fonts, larger subscript/superscript text),
    so they pass every typography-based heading check even though they're
    pure noise — e.g. "Q Au =", "fl fl", "2 O SH", "( RV L \u03b7". This check
    catches what typography can't: whether the line actually reads as a
    title/phrase rather than equation residue, a URL, or a metadata stamp.
    """
    stripped = text.strip()

    # URLs are never headings.
    if re.search(r"https?://|www\.", stripped, re.IGNORECASE):
        return False
    # "Label: number" metadata stamps (Eprints ID: 5819, DOI: 10.1234, ...)
    if re.match(r"^\s*[A-Za-z ]+:\s*\d", stripped):
        return False

    letters_only = re.sub(r"[^a-zA-Z]", "", stripped)
    if len(letters_only) < 3:
        return False  # almost no actual letters — pure symbols/numbers

    # Real words (3+ consecutive letters) vs. symbol/variable-name residue
    # (single/double letters like "Q", "Au", "fl", "SH" mixed with math).
    tokens = re.findall(r"[A-Za-z]+", stripped)
    real_word_tokens = [t for t in tokens if len(t) >= 3]
    if not real_word_tokens:
        return False  # every alphabetic token is 1-2 letters

    real_word_chars = sum(len(t) for t in real_word_tokens)
    if real_word_chars / max(len(letters_only), 1) < 0.6:
        return False  # mostly symbol residue, only incidentally has a real word

    # Headings are title/phrase fragments, not clause fragments — a comma
    # usually signals a body-text line that landed in a heading-sized font
    # by accident (e.g. inline math/citation), not an actual section title.
    if "," in stripped:
        return False

    # A real heading/title starts with a capital letter or a digit (numbered
    # heading); body-text fragments that accidentally got a heading-sized
    # font (often from italic math variables or citation snippets) almost
    # always start mid-clause with a lowercase word ("and λ", "with",
    # "represents the", "value of C") or with leading punctuation like "&"
    # or "(" ("& Mathies 2003"). Confirmed this exact failure mode in real
    # output: dozens of these fragments were passing as level-2 headings
    # and getting concatenated onto a paper's title, which then never got
    # reset for the rest of the document.
    first_char = stripped[0] if stripped else ""
    if not (first_char.isupper() or first_char.isdigit()):
        return False

    if re.fullmatch(r"[IVXLCDM]+(\s+[IVXLCDM]+)*", stripped):
        return False

    return True


def detect_running_headers(all_line_texts: list[str], total_pages: int, min_repeat_frac: float = 0.4) -> set[str]:
    """
    A line that recurs identically across many pages (e.g. "ARTICLE IN
    PRESS" printed as a running header/footer on every page, or a journal
    name + volume number repeated in a margin) is not real content, no
    matter what font size it's in — running headers/footers are sometimes
    *smaller* than body text, not larger, so this must scan every extracted
    line rather than only typography-flagged heading candidates. A real
    section heading should never appear verbatim on a large fraction of a
    paper's pages, so a high repeat threshold (default 40% of pages) is a
    safe way to single these out without catching genuinely recurring
    short phrases that happen to appear in body text once or twice.
    """
    if total_pages <= 0:
        return set()
    counts = Counter(t for t in all_line_texts if t.strip())
    threshold = max(3, round(total_pages * min_repeat_frac))
    return {text for text, n in counts.items() if n >= threshold and len(text.strip()) < 80}


def extract_lines_with_style(pdf_path: str) -> list[dict]:
    """
    Extract text line-by-line with font size/weight metadata, by grouping
    pdfplumber's word-level output into visual lines (words at the same
    vertical position). This is what lets us tell headings apart from body
    text by *appearance* rather than by guessing vocabulary.

    Returns a list of {"page": int, "text": str, "size": float, "bold": bool}.
    """
    lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # x_tolerance_ratio (relative to font size) instead of a fixed
            # x_tolerance: pdfplumber's default x_tolerance=3 is an absolute
            # point value, but real inter-word gaps shrink with font size.
            # Dense academic PDFs with small body text (8-9pt) or tight
            # kerning can have actual word gaps narrower than that fixed
            # default, causing entire runs of words to merge into one
            # "word" token with no space at all (confirmed: reproduced this
            # exact failure with a synthetic tightly-kerned PDF, where
            # default x_tolerance merged a whole sentence into one token).
            words = page.extract_words(extra_attrs=["size", "fontname"], x_tolerance_ratio=0.2)
            if not words:
                continue
            current_line, current_top = [], None
            for w in words:
                if current_top is None or abs(w["top"] - current_top) < 3:
                    current_line.append(w)
                    current_top = w["top"] if current_top is None else current_top
                else:
                    lines.append(_line_from_words(page_num, current_line))
                    current_line, current_top = [w], w["top"]
            if current_line:
                lines.append(_line_from_words(page_num, current_line))
    return lines


def _line_from_words(page_num: int, words: list[dict]) -> dict:
    text = " ".join(w["text"] for w in words)
    text = strip_page_header_prefix(text)
    text = fix_cid_ligatures(text)
    text = repair_concatenated_words(text)
    avg_size = sum(w["size"] for w in words) / len(words)
    is_bold = any("bold" in w["fontname"].lower() for w in words)
    return {"page": page_num, "text": text, "size": avg_size, "bold": is_bold}


def clean_text(text: str) -> str:
    """De-hyphenate line-wrap artifacts and collapse whitespace."""
    text = re.sub(r"-\s*$", "", text)  # trailing hyphen from line wrap
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_short_noise_fragment(text: str) -> bool:
    """
    Catch equation/symbol residue lines that are short enough to be
    standalone noise (e.g. "Q Au =", "fl fl", "2 O SH") so they don't get
    glued into body prose as if they were a real sentence — these get
    rejected as headings by looks_like_real_heading_text, but without this
    check they'd otherwise fall through into the body-text bucket since
    they're also not bylines. Deliberately scoped to SHORT lines only
    (<= 8 words) so it can never accidentally eat a real, longer body
    sentence that happens to contain some inline math or symbols.
    """
    stripped = text.strip()
    if len(stripped.split()) > 8:
        return False  # too long to be pure equation residue; let it through
    return not looks_like_real_heading_text(stripped) and not stripped.endswith(".")


def looks_like_heading(line: dict, body_size: float, heading_size_mode: float | None = None) -> tuple[bool, int]:
    """
    Decide whether a line is a heading, and at what nesting level
    (1 = top-level section, 2 = sub-heading), using typography first and
    keyword/numbering patterns as secondary support.

    heading_size_mode, if provided, is the most common font size among
    *other* detected headings in this document — used to tell a genuine
    sub-heading (smaller than typical headings) apart from a one-off larger
    title/byline line, rather than comparing against a fixed multiplier
    that breaks on documents where the title happens to be only slightly
    larger than section headings.

    Returns (is_heading, level).
    """
    text = line["text"].strip()
    word_count = len(text.split())

    if not text or word_count > MAX_HEADING_WORDS:
        return False, 0
    # Headings essentially never end in a period (body sentences do).
    if text.endswith("."):
        return False, 0
    # Reject title-page noise (author bylines, affiliations, page numbers)
    # BEFORE checking typography — these often share the same large/bold
    # font as real headings, so size alone can't distinguish them.
    if looks_like_byline_or_junk(text):
        return False, 0
    # Reject math/equation fragments, URLs, and metadata stamps — these are
    # routinely rendered in heading-sized/styled fonts (inline math uses
    # italic or symbol fonts that are often larger) but contain no actual
    # heading-like words ("Q Au =", "fl fl", "2 O SH").
    if not looks_like_real_heading_text(text):
        return False, 0

    size_ratio = line["size"] / body_size if body_size else 1.0
    numbering_match = NUMBERED_HEADING_PREFIX.match(text)
    is_numbered = bool(numbering_match)
    is_known_keyword = bool(KNOWN_HEADING_WORDS.search(text)) and word_count <= 4

    def assign_level() -> int:
        # Numbering depth is the strongest, least brittle signal: "2." is
        # a top-level section, "2.1" is a sub-heading, regardless of font
        # size quirks. Depth comes from the number of numeric segments in
        # the captured prefix itself ("2" -> 1, "2.1" -> 2, "2.1.3" -> 3),
        # not from counting periods in the whole line — "2." and "2.1" both
        # contain exactly one period, but represent different depths.
        if is_numbered:
            numeric_prefix = numbering_match.group(1)
            depth = numeric_prefix.count(".") + 1
            return 1 if depth <= 1 else 2
        # Otherwise, compare this line's size against the typical size of
        # OTHER headings already seen in this document (not a fixed
        # multiplier) — meaningfully smaller than that typical size means
        # "sub-heading"; anything else (including one-off larger titles)
        # defaults to level 1, since over-nesting is worse for retrieval
        # than under-nesting (it buries real sections under a parent label).
        if heading_size_mode and line["size"] < heading_size_mode * 0.95:
            return 2
        return 1

    # Primary signal: visibly larger and/or bold relative to body text.
    if size_ratio >= HEADING_SIZE_RATIO or (line["bold"] and size_ratio >= 1.0):
        return True, assign_level()

    # Secondary signal: numbered prefix ("2.1 Capillary-Based Reactors") even
    # if the font-size jump is small — numbering itself is a strong heading cue.
    if is_numbered and word_count <= MAX_HEADING_WORDS:
        return True, assign_level()

    # Tertiary fallback: known section keyword, short line, same style as body.
    # Catches PDFs where headings aren't styled distinctly at all.
    if is_known_keyword:
        return True, 1

    return False, 0


def split_into_sections(lines: list[dict]) -> list[dict]:
    """
    Walk the line-level extraction and group lines into sections using the
    typography-based heading detector. Tracks heading hierarchy so a
    sub-heading's text is still tagged with its parent section name.

    Returns a list of {"section": str, "text": str} dicts. If no headings
    are detected at all, returns a single section per ~page-worth of text
    instead of one giant "Body" blob (better for review papers / PDFs
    without font-size variation).
    """
    if not lines:
        return []

    sizes = [l["size"] for l in lines]
    # Body text size = the most common (mode-like) size, robust to a few
    # large headings skewing a simple average.
    body_size = statistics.median(sizes)
    total_pages = max((l["page"] for l in lines), default=1)

    # Pre-pass: scan EVERY line (not just heading-shaped ones) for text that
    # recurs across many pages — running headers/footers are sometimes
    # printed smaller than body text, so they wouldn't all show up as
    # heading candidates and need this separate, broader pass.
    all_line_texts = [line["text"].strip() for line in lines]
    running_headers = detect_running_headers(all_line_texts, total_pages)

    # Second pre-pass: find the most common font size among heading
    # candidates (using a provisional, mode-less pass) so the real pass
    # below can tell "smaller than typical headings -> sub-heading" apart
    # from "larger one-off title -> still level 1" without relying on a
    # brittle fixed multiplier that breaks when a title and its sections
    # happen to be similarly sized.
    provisional_heading_sizes = [
        line["size"] for line in lines
        if line["text"].strip() not in running_headers and looks_like_heading(line, body_size)[0]
    ]
    # Only trust a "typical heading size" if some size actually repeats —
    # a single title line is a poor stand-in for "the typical heading size"
    # and would otherwise make every same-or-smaller real section heading
    # look like a sub-heading by comparison (statistics.mode on a tie of
    # unique values just returns whichever appeared first, which is often
    # the title itself in a short/sparse document).
    size_counts = Counter(provisional_heading_sizes)
    repeated_sizes = [size for size, n in size_counts.items() if n >= 2]
    heading_size_mode = statistics.mode(repeated_sizes) if repeated_sizes else None

    sections = []
    current_h1, current_h2 = "Front Matter", None
    current_text_parts = []

    def flush():
        if current_text_parts:
            name = current_h1 if not current_h2 else f"{current_h1} — {current_h2}"
            text = clean_text(" ".join(current_text_parts))
            if text:  # keep any non-empty section; chunk-level filtering happens later
                sections.append({"section": name, "text": text})

    any_heading_found = False
    prev_heading_level = None  # level of the immediately preceding line, if it was a heading
    prev_heading_size = None   # font size of that preceding heading line
    for line in lines:
        line_text = line["text"].strip()
        if line_text in running_headers:
            continue  # journal running header/footer repeated on every page — not content
        is_heading, level = looks_like_heading(line, body_size, heading_size_mode)
        if is_heading:
            any_heading_found = True
            # Merge with the previous heading only if it's the SAME level
            # AND the SAME font size — wrapped lines of one multi-line
            # title/heading are rendered at a single consistent size, so
            # this distinguishes that case from two genuinely different,
            # back-to-back headings (e.g. a title immediately followed by
            # "Introduction" in a different, smaller size) which should
            # stay separate, not get concatenated into one label.
            same_size = prev_heading_size is not None and abs(line["size"] - prev_heading_size) < 0.5
            if prev_heading_level == level and same_size:
                if level == 1:
                    current_h1 = f"{current_h1} {line_text}".strip()
                else:
                    current_h2 = f"{current_h2} {line_text}".strip()
            else:
                flush()
                current_text_parts = []
                if level == 1:
                    current_h1, current_h2 = line_text, None
                else:
                    current_h2 = line_text
            prev_heading_level, prev_heading_size = level, line["size"]
        elif looks_like_byline_or_junk(line["text"]):
            prev_heading_level = prev_heading_size = None
            continue  # author bylines/affiliations/page numbers: drop, don't keep as body prose
        elif _is_short_noise_fragment(line["text"]):
            prev_heading_level = prev_heading_size = None
            continue  # equation/symbol residue short enough to be noise, not a real sentence
        else:
            prev_heading_level = prev_heading_size = None
            current_text_parts.append(line["text"])
    flush()

    if not any_heading_found:
        # No typographic or keyword headings detected anywhere (flat-format
        # review paper, or a PDF that lost font metadata). Don't collapse to
        # one "Body" blob — that starves chunking of any topical labeling.
        # Instead, fall back to splitting by page, which at least gives the
        # QA generator a rough positional anchor ("early in the paper" vs
        # "late in the paper") rather than nothing.
        full_text = clean_text(" ".join(l["text"] for l in lines))
        return [{"section": "Body (no headings detected)", "text": full_text}]

    # Drop excluded sections (References/Bibliography/Acknowledgments) BEFORE
    # merging tiny sections together — otherwise a short "Methods" section
    # could get merged with an adjacent "References" section into a combined
    # label like "Methods / References", and a label-based exclusion check
    # run after merging would then discard the legitimate Methods content
    # along with it. Filtering first means merging only ever sees sections
    # that are supposed to survive.
    sections = [s for s in sections if not EXCLUDED_SECTION_KEYWORDS.search(s["section"])]

    return _merge_tiny_sections(sections)


def _merge_tiny_sections(sections: list[dict], min_merge_chars: int = 150) -> list[dict]:
    """
    Heavily sub-headed papers (e.g. each sub-heading covering just one or two
    sentences) produce many tiny sections in a row. A tiny standalone chunk
    gives the QA generator too little to work with, so merge runs of small
    *consecutive* sections together, keeping a combined section label
    (e.g. "Closed-Channel Microreactors — 2.1 ... / 2.2 ...").
    """
    if not sections:
        return sections

    merged = []
    buffer_names, buffer_text = [], ""
    for sec in sections:
        if len(buffer_text) + len(sec["text"]) < min_merge_chars * 2:
            buffer_names.append(sec["section"])
            buffer_text = (buffer_text + " " + sec["text"]).strip()
        else:
            if buffer_text:
                merged.append({"section": " / ".join(dict.fromkeys(buffer_names)), "text": buffer_text})
            buffer_names, buffer_text = [sec["section"]], sec["text"]
    if buffer_text:
        merged.append({"section": " / ".join(dict.fromkeys(buffer_names)), "text": buffer_text})
    return merged


def sliding_window_chunks(text: str, chunk_chars: int, overlap_chars: int) -> list[str]:
    """Split text into overlapping fixed-size windows, breaking on sentence
    boundaries where possible so chunks don't cut mid-sentence."""
    if len(text) <= chunk_chars:
        return [text] if len(text) >= MIN_CHUNK_CHARS else []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) <= chunk_chars:
            current += (" " if current else "") + sent
        else:
            if len(current) >= MIN_CHUNK_CHARS:
                chunks.append(current.strip())
            overlap_text = current[-overlap_chars:] if current else ""
            current = (overlap_text + " " + sent).strip()
    if len(current) >= MIN_CHUNK_CHARS:
        chunks.append(current.strip())
    return chunks


def chunk_paper(pdf_path: str) -> list[dict]:
    """
    Full pipeline for one PDF: extract lines with style -> detect headings
    by typography -> group into sections (with sub-heading tracking) ->
    window-chunk each section. Returns a list of chunk dicts ready to write
    to JSONL.
    """
    paper_title = Path(pdf_path).stem
    lines = extract_lines_with_style(pdf_path)

    if not lines:
        return []  # scanned PDF with no text layer — needs OCR, see README

    sections = split_into_sections(lines)

    chunks = []
    for sec in sections:
        for window in sliding_window_chunks(sec["text"], CHUNK_CHARS, OVERLAP_CHARS):
            chunks.append(
                {
                    "chunk_id": str(uuid.uuid4())[:8],
                    "paper_title": paper_title,
                    "section": sec["section"],
                    "text": window,
                    "source_file": os.path.basename(pdf_path),
                }
            )
    return chunks


def main():
    pdf_dir = Path(PDF_DIR)
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {PDF_DIR}. Set PDF_DIR to your papers folder.")
        return

    all_chunks = []
    for pdf_path in pdf_files:
        print(f"Processing {pdf_path.name} ...")
        try:
            chunks = chunk_paper(str(pdf_path))
            print(f"  -> {len(chunks)} chunks")
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  !! failed: {e}")

    with open(OUT_PATH, "w") as f:
        for c in all_chunks:
            f.write(json.dumps(c) + "\n")

    print(f"\nWrote {len(all_chunks)} chunks from {len(pdf_files)} papers to {OUT_PATH}")
    print("Sanity check a few chunks:")
    for c in all_chunks[:2]:
        print(json.dumps(c, indent=2)[:500])


if __name__ == "__main__":
    main()