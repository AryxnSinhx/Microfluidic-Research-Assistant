"""
Gradio demo: Microfluidics + Bilirubin Detection RAFT Research Assistant.

Run this in the same Kaggle notebook/environment as ablation_study.ipynb -- it
reuses the identical validated pipeline: fp16 merged model, no quantization,
all-mpnet-base-v2 retrieval, the exact [DOC N] (paper_title, section) text
prompt format, greedy generation with repetition_penalty=1.1, the 0.4
retrieval-score abstention gate, and post-hoc repeat-sentence truncation.

NEW IN THIS VERSION
--------------------
1. Live PDF ingestion: upload a paper mid-session, it's chunked, embedded,
   and added to a SEPARATE session-scoped FAISS index (your validated
   903-chunk core index is never mutated). Retrieval merges both indexes
   at query time and tags each hit with its source ("core corpus" vs
   "uploaded: filename.pdf") so you can see in the UI exactly where an
   answer's grounding came from.
2. UI upgrade: two-tab layout (Ask / Manage Corpus), styled source badges,
   a live "corpus status" panel, upload progress feedback, and a session
   reset button.

IMPORTANT CAVEAT (carried over from your open item #1)
--------------------------------------------------------
Your false-abstention bug (all raft_domain abstention examples sharing one
template) means a freshly uploaded paper is exactly the scenario that can
trigger a false "this isn't in the corpus" response even when retrieval
found it. The UI surfaces the raw retrieval score for every hit so you can
tell apart "genuinely low similarity" from "the generator is abstaining
even though DOC 1 is clearly the uploaded paper."

USAGE:
    !pip install gradio faiss-cpu pdfplumber -q
    # then run this script, or paste into a notebook cell
"""

import gc
import os
import re
import pickle

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gradio as gr
import numpy as np
import torch
import faiss
import pdfplumber
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# %% CONFIG -- identical to ablation_study.py's validated settings ------

FT_MODEL_ID = "aryxnsinhx/mistral-7b-microfluidics-raft"
DATASET_DIR = "/kaggle/input/datasets/aryxnsinhx1/ablation-aryxn"
INDEX_PATH = f"{DATASET_DIR}/microfluidics.index"
METADATA_PATH = f"{DATASET_DIR}/chunk_metadata.pkl"

RETRIEVAL_SCORE_THRESHOLD = 0.4
TOP_K = 5
MAX_CHUNK_CHARS_IN_PROMPT = 700

GEN_KWARGS = dict(
    do_sample=False,
    repetition_penalty=1.1,
    max_new_tokens=350,
)

# --- New: session ingestion config --------------------------------------
# NOTE: these should mirror the chunking parameters used to build your
# core 903-chunk corpus. Adjust to match if your original chunker used
# different sizes -- consistency here is what keeps retrieval quality
# even between core and uploaded content (see our chunking discussion).
SESSION_CHUNK_SIZE_CHARS = 900
SESSION_CHUNK_OVERLAP_CHARS = 150
MAX_SESSION_PAPERS = 8
MAX_SESSION_CHUNKS = 400  # safety cap so a huge PDF doesn't blow up memory

EMBED_DIM = 768  # all-mpnet-base-v2

# %% LOAD -- runs once at startup ----------------------------------------

print("Loading embedder + core FAISS index...")
embedder = SentenceTransformer("all-mpnet-base-v2", device="cpu")
core_index = faiss.read_index(INDEX_PATH)
with open(METADATA_PATH, "rb") as f:
    raw = pickle.load(f)
core_chunks = raw.to_dict(orient="records") if hasattr(raw, "to_dict") else list(raw)
print(f"Loaded {core_index.ntotal} core vectors, {len(core_chunks)} chunk records.")

print("Loading fine-tuned model (fp16, no quantization)...")
tokenizer = AutoTokenizer.from_pretrained(FT_MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    FT_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
    attn_implementation="sdpa",
)
model.eval()
print("Model loaded.")


# %% INGESTION -- new: PDF -> chunks -> session FAISS index --------------

def extract_pdf_text(filepath):
    """Extract text page by page, keeping page boundaries for section labels."""
    pages = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append(text)
    return pages


def guess_section_label(paragraph, page_num):
    """Light heuristic: use a header-like line if present, else fall back
    to a page reference. This is a starting point -- if your core corpus
    used a more rigorous section-detection step, swap it in here so
    session chunks carry the same style of section metadata."""
    first_line = paragraph.strip().split("\n", 1)[0].strip()
    looks_like_header = (
        0 < len(first_line) < 80
        and (first_line.isupper() or re.match(r"^\d+(\.\d+)*\.?\s+[A-Z]", first_line))
    )
    if looks_like_header:
        return first_line
    return f"page {page_num}"


def chunk_pdf_pages(pages, chunk_size=SESSION_CHUNK_SIZE_CHARS, overlap=SESSION_CHUNK_OVERLAP_CHARS):
    """Paragraph-aware chunking with overlap, mirroring the guidance we
    discussed: split on natural paragraph boundaries first, then pack into
    overlapping windows so information near a boundary isn't orphaned."""
    chunks = []
    for page_num, page_text in enumerate(pages, start=1):
        if not page_text.strip():
            continue
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", page_text) if p.strip()]
        buf = ""
        for para in paragraphs:
            candidate = (buf + "\n\n" + para).strip() if buf else para
            if len(candidate) <= chunk_size:
                buf = candidate
                continue
            if buf:
                chunks.append((buf, guess_section_label(buf, page_num)))
                buf = buf[-overlap:] + "\n\n" + para if overlap else para
            else:
                # single paragraph longer than chunk_size: hard-split it
                for i in range(0, len(para), chunk_size - overlap):
                    piece = para[i:i + chunk_size]
                    chunks.append((piece, guess_section_label(piece, page_num)))
                buf = ""
        if buf:
            chunks.append((buf, guess_section_label(buf, page_num)))
    return chunks


def ingest_pdf(filepath, paper_title, session_state):
    session_state = session_state or {"index": None, "chunks": [], "papers": []}

    if len(session_state["papers"]) >= MAX_SESSION_PAPERS:
        return session_state, (
            f"Session cap reached ({MAX_SESSION_PAPERS} papers). "
            "Reset the session to ingest more."
        )

    try:
        pages = extract_pdf_text(filepath)
    except Exception as e:
        return session_state, f"Failed to read PDF: {e}"

    raw_chunks = chunk_pdf_pages(pages)
    if not raw_chunks:
        return session_state, "No extractable text found in this PDF (is it scanned/image-only?)."

    remaining_budget = MAX_SESSION_CHUNKS - len(session_state["chunks"])
    if remaining_budget <= 0:
        return session_state, f"Session chunk cap reached ({MAX_SESSION_CHUNKS}). Reset to continue."
    raw_chunks = raw_chunks[:remaining_budget]

    texts = [c[0] for c in raw_chunks]
    embs = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    embs = np.array(embs, dtype="float32")

    if session_state["index"] is None:
        session_state["index"] = faiss.IndexFlatIP(EMBED_DIM)

    session_state["index"].add(embs)
    for text, section in raw_chunks:
        session_state["chunks"].append({
            "paper_title": paper_title,
            "section": section,
            "text": text,
        })
    session_state["papers"].append({"title": paper_title, "n_chunks": len(raw_chunks)})

    status = (
        f"Ingested \"{paper_title}\": {len(raw_chunks)} chunks added. "
        f"Session total: {len(session_state['chunks'])} chunks across "
        f"{len(session_state['papers'])} paper(s)."
    )
    return session_state, status


def corpus_status_md(session_state):
    if not session_state or not session_state.get("papers"):
        return "_No papers uploaded this session. Answers are grounded in the core corpus only._"
    lines = ["**Session-uploaded papers (this browser session only):**\n"]
    for p in session_state["papers"]:
        lines.append(f"- {p['title']} — {p['n_chunks']} chunks")
    return "\n".join(lines)


def reset_session(session_state):
    return {"index": None, "chunks": [], "papers": []}, "_Session cleared. Core corpus only._", ""


# %% RETRIEVAL / GENERATION -- core logic unchanged, now source-merged --

def retrieve(query, session_state, top_k=TOP_K):
    q_emb = np.array(embedder.encode([query], normalize_embeddings=True), dtype="float32")

    hits = []

    scores, idxs = core_index.search(q_emb, top_k)
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        c = core_chunks[idx]
        hits.append({
            "score": float(score),
            "paper_title": c["paper_title"],
            "section": c["section"],
            "text": c["text"],
            "source": "core corpus",
        })

    if session_state and session_state.get("index") is not None and session_state["index"].ntotal > 0:
        s_scores, s_idxs = session_state["index"].search(q_emb, top_k)
        for score, idx in zip(s_scores[0], s_idxs[0]):
            if idx == -1:
                continue
            c = session_state["chunks"][idx]
            hits.append({
                "score": float(score),
                "paper_title": c["paper_title"],
                "section": c["section"],
                "text": c["text"],
                "source": f"uploaded: {c['paper_title']}",
            })

    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits[:top_k]


def format_doc_block(hits):
    blocks = []
    for i, h in enumerate(hits, start=1):
        text = h["text"]
        if len(text) > MAX_CHUNK_CHARS_IN_PROMPT:
            text = text[:MAX_CHUNK_CHARS_IN_PROMPT].rsplit(" ", 1)[0] + "..."
        blocks.append(f"[DOC {i}] ({h['paper_title']}, {h['section']}) {text}")
    return "\n\n".join(blocks)


def build_prompt(question, doc_block):
    return (
        "You are a research assistant. Use only the context below to answer. "
        "If the context does not contain the answer, say you don't know.\n\n"
        f"Context:\n{doc_block}\n\nQuestion: {question}\n[/INST]"
    )


def truncate_at_first_repeat(text):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    seen, kept = set(), []
    for s in sentences:
        k = s.strip().lower()
        if k in seen and len(k) > 15:
            break
        seen.add(k)
        kept.append(s)
    return " ".join(kept)


def generate(prompt):
    inputs = None
    out = None
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, **GEN_KWARGS, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return truncate_at_first_repeat(text), None
    except torch.cuda.OutOfMemoryError as e:
        return None, f"OOM: {e}"
    finally:
        del inputs, out
        gc.collect()
        torch.cuda.empty_cache()


# %% DEMO LOGIC -----------------------------------------------------------

SOURCE_BADGE_CSS = {
    "core": "background:#2D3A4A; color:#9FD3FF;",
    "upload": "background:#3A2D4A; color:#D9B8FF;",
}


def badge_for(source):
    style = SOURCE_BADGE_CSS["upload"] if source.startswith("uploaded:") else SOURCE_BADGE_CSS["core"]
    return f'<span style="{style} padding:2px 8px; border-radius:10px; font-size:11px;">{source}</span>'


def answer_question(question, session_state):
    """Core retrieve-then-generate call. Returns (answer_text, citation_markdown)."""
    if not question or not question.strip():
        return "", ""

    hits = retrieve(question, session_state)
    top_score = hits[0]["score"] if hits else 0.0

    if top_score < RETRIEVAL_SCORE_THRESHOLD:
        answer = (
            f"I don't have a confident enough match in the corpus to answer this "
            f"reliably. (Top retrieval score: {top_score:.2f}, below the "
            f"{RETRIEVAL_SCORE_THRESHOLD:.2f} threshold.)"
        )
        citation_md = "_No sources met the confidence threshold for this query._"
        return answer, citation_md

    doc_block = format_doc_block(hits)
    prompt = build_prompt(question, doc_block)
    answer, err = generate(prompt)
    if err:
        answer = f"Generation failed: {err}"

    citation_lines = ["**Sources retrieved:**"]
    for i, h in enumerate(hits, start=1):
        citation_lines.append(
            f"{i}. {badge_for(h['source'])} *{h['paper_title']}* — {h['section']} "
            f"(similarity: {h['score']:.2f})"
        )
    citation_md = "\n".join(citation_lines)

    return answer, citation_md


EXAMPLES = [
    "What physical mechanism drives mixing in a microfluidic channel at low Reynolds number?",
    "What causes electrohydrodynamic instability at a liquid-liquid interface under an applied electric field?",
    "How is a diazonium salt used to detect bilirubin in a sample, chemically speaking?",
    "Explain the Van den Bergh reaction and how it distinguishes conjugated from unconjugated bilirubin.",
]

def latest_sources_md(session_state, last_hits):
    """Right-panel content: sources for the most recent answer only,
    rendered as cards rather than buried in the chat transcript."""
    if not last_hits:
        return "_Ask a question to see grounding sources here._"
    lines = []
    for i, h in enumerate(last_hits, start=1):
        lines.append(
            f"**{i}.** {badge_for(h['source'])}<br>"
            f"*{h['paper_title']}* — {h['section']}<br>"
            f"similarity: `{h['score']:.2f}`\n"
        )
    return "\n\n".join(lines)


def chat_respond_v2(question, chat_history, session_state):
    """Same as chat_respond but also returns the raw hit list so the
    right-hand sources panel can render independently of the chat bubble."""
    chat_history = chat_history or []
    if not question or not question.strip():
        return chat_history, "", session_state, latest_sources_md(session_state, None)

    hits = retrieve(question, session_state)
    top_score = hits[0]["score"] if hits else 0.0

    if top_score < RETRIEVAL_SCORE_THRESHOLD:
        answer = (
            f"I don't have a confident enough match in the corpus to answer this "
            f"reliably. (Top retrieval score: {top_score:.2f}, below the "
            f"{RETRIEVAL_SCORE_THRESHOLD:.2f} threshold.)"
        )
        hits_for_panel = hits
    else:
        doc_block = format_doc_block(hits)
        prompt = build_prompt(question, doc_block)
        answer, err = generate(prompt)
        if err:
            answer = f"Generation failed: {err}"
        hits_for_panel = hits

    chat_history = chat_history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    return chat_history, "", session_state, latest_sources_md(session_state, hits_for_panel)


CUSTOM_CSS = """
html, body { height: 100vh !important; overflow: hidden !important; }
.gradio-container { max-width: 1280px !important; height: 100vh !important; margin: auto !important; padding-top: 8px !important; overflow: hidden !important; }
#header-bar { display: flex !important; align-items: center !important; justify-content: space-between !important; margin-bottom: 0 !important; }
#domain-line p { margin: 2px 0 !important; font-size: 13px !important; line-height: 1.4 !important; }
#domain-line h2 { margin-bottom: 2px !important; font-size: 28px !important; }
#corpus-pill { font-size: 12px !important; padding: 3px 10px !important; border-radius: 12px !important; background: rgba(120,120,120,0.12) !important; }
.accordion { margin: 4px 0 !important; }
#chatbot { height: 46vh !important; }
#chatbot .message { font-family: Georgia, serif !important; font-size: 14px !important; line-height: 1.45 !important; }
#sources-panel { border-left: 1px solid rgba(120,120,120,0.25); padding-left: 18px !important; font-size: 12px !important; max-height: 46vh !important; overflow-y: auto !important; }
#sources-title { font-weight: 600 !important; margin-bottom: 6px !important; }
footer { display: none !important; }
"""

with gr.Blocks(css=CUSTOM_CSS, title="Microfluidics RAFT Research Assistant", theme=gr.themes.Soft()) as demo:
    session_state = gr.State({"index": None, "chunks": [], "papers": []})

    # ---- Top bar: title + live corpus status + collapsed upload -------
    with gr.Row(elem_id="header-bar"):
        gr.Markdown(
            "## Microfluidics & Bilirubin Detection Research Assistant\n"
            "RAFT-tuned Mistral-7B + retrieval, with live paper uploads layered in per session.\n\n"
            "**Domain expertise:** microfluidic mixing & electrohydrodynamic instability · "
            "Micro-channel device fabrication · microfluidic reaction kinetics · Complex microfluidic system · "
            "related fluid dynamics & electrochemistry literature",
            elem_id="domain-line",
        )
    corpus_status = gr.Markdown(corpus_status_md(None), elem_id="corpus-pill")

    with gr.Accordion("Manage session corpus (upload a paper / reset)", open=False):
        with gr.Row():
            with gr.Column(scale=2):
                pdf_upload = gr.File(label="PDF file", file_types=[".pdf"])
                paper_title_box = gr.Textbox(
                    label="Paper title (used in citations)",
                    placeholder="e.g. Smith et al. 2023, Microchannel Mixing",
                )
                with gr.Row():
                    ingest_btn = gr.Button("Ingest into session corpus", variant="primary", size="sm")
                    reset_btn = gr.Button("Reset session", variant="stop", size="sm")
                upload_status = gr.Markdown(elem_id="upload-status")
            with gr.Column(scale=1):
                gr.Markdown(
                    "Uploaded PDFs are chunked/embedded the same way as the core "
                    "903-chunk corpus and added to a **session-only** index — the "
                    "validated core index is never modified. Sources are labeled "
                    "distinctly in the panel to the right of the chat."
                )

    # ---- Main row: chat (wide) + live sources panel (narrow) ----------
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label=None, elem_id="chatbot", type="messages",
                show_label=False, avatar_images=(None, None),
            )
            with gr.Row():
                question_box = gr.Textbox(
                    label=None, show_label=False,
                    placeholder="Ask a question about the corpus...",
                    scale=5, container=False,
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)

        with gr.Column(scale=1, min_width=260, elem_id="sources-panel"):
            gr.Markdown("**Sources for last answer**", elem_id="sources-title")
            sources_panel = gr.Markdown(latest_sources_md(None, None))

    # --- wiring ---
    submit_btn.click(
        chat_respond_v2,
        inputs=[question_box, chatbot, session_state],
        outputs=[chatbot, question_box, session_state, sources_panel],
    )
    question_box.submit(
        chat_respond_v2,
        inputs=[question_box, chatbot, session_state],
        outputs=[chatbot, question_box, session_state, sources_panel],
    )

    def _ingest_and_report(file, title, state):
        if file is None:
            return state, "Please choose a PDF first.", corpus_status_md(state)
        title = title.strip() or os.path.basename(file.name)
        new_state, status = ingest_pdf(file.name, title, state)
        return new_state, status, corpus_status_md(new_state)

    ingest_btn.click(
        _ingest_and_report,
        inputs=[pdf_upload, paper_title_box, session_state],
        outputs=[session_state, upload_status, corpus_status],
    )

    reset_btn.click(reset_session, inputs=[session_state], outputs=[session_state, corpus_status, upload_status])

if __name__ == "__main__":
    demo.launch(share=True)