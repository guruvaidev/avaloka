"""
Daft Documentation Ingestion Script — Markdown Version
=======================================================
Reads converted .md files, chunks by function/method boundary,
and stores in a Chroma vector database.

Chunking Strategy (one function = one chunk):
─────────────────────────────────────────────
Single-function files  (e.g. any_value.md, approx_count_distinct.md)
  → One ## heading per file → one chunk

Class/module method pages  (e.g. aggregations.md with GroupedDataFrame)
  → One ## class header + multiple ### method headings
  → n_h3 >= 3 triggers ### split → one chunk per method

Big reference pages  (e.g. dataframe.md with many ## functions)
  → Multiple ## headings → one chunk per ## section

Routing rules (in priority order):
  1. n_h3 >= 3  →  split on ###   (class method pages)
  2. n_h2 >= 1  →  split on ##    (single or multi-function pages)
  3. n_h3 >= 1  →  split on ###   (rare: only ### headings)
  4. fallback   →  whole file as one chunk

Footer noise stripped before chunking:
  "Back to top", "© Copyright", "Previous/Next" nav links, scarf pixel
"""

import re
from pathlib import Path
from typing import List, Dict, Optional
import chromadb
from chromadb.utils import embedding_functions


# ── Configuration ─────────────────────────────────────────────────────────────
DOCS_MD_PATH    = "rag_docs/markdown"    # relative to project root (2 levels up)
VECTOR_DB_PATH  = "daft_embeddings"
COLLECTION_NAME = "daft_documentation"
EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"

MAX_CHUNK_CHARS = 4000   # hard cap; oversized chunks split at paragraph boundaries
MIN_CHUNK_CHARS = 100    # discard fragments shorter than this


# ── Footer noise ──────────────────────────────────────────────────────────────
_NOISE_PATTERNS = [
    r"Back to top.*",
    r"© Copyright \d{4},.*",
    r"Previous\s*\n.*?\n.*?Next.*",              # nav block
    r"\[Previous[^\]]*\]\([^)]*\)",              # [Previous text](url)
    r"\[Next[^\]]*\]\([^)]*\)",                  # [Next text](url)
    r"!\[\]\(https://static\.scarf\.sh[^)]*\)",  # tracking pixel
    r"Source code in `[^`]+`",                   # source label
    r"-{3,}",                                    # horizontal rules
]
# NOTE: no re.DOTALL — with it, `Back to top.*` etc. match `.` across newlines
# to end-of-file and delete every later section. The one multi-line pattern
# (the Previous/Next nav block) uses explicit `\n`, so it still matches.
_NOISE_RE = re.compile("|".join(_NOISE_PATTERNS), re.MULTILINE)


# ── Category inference ────────────────────────────────────────────────────────
_CATEGORY_MAP = {
    "aggregat":   "Aggregation",
    "groupby":    "Aggregation",
    "filter":     "Filtering",
    "select":     "Selection",
    "sort":       "Sorting",
    "join":       "Joins",
    "read_":      "IO",
    "write_":     "IO",
    "udf":        "UDF",
    "func":       "UDF",
    "expression": "Expressions",
    "datatype":   "DataTypes",
    "schema":     "Schema",
    "series":     "Series",
    "dataframe":  "DataFrame",
    "window":     "Window",
    "string":     "StringOps",
    "datetime":   "DateTimeOps",
    "null":       "NullHandling",
}

def _infer_category(filename: str, heading: str) -> str:
    combined = (filename + " " + heading).lower()
    for kw, cat in _CATEGORY_MAP.items():
        if kw in combined:
            return cat
    return "General"


# ── Text helpers ──────────────────────────────────────────────────────────────

def _strip_noise(text: str) -> str:
    text = _NOISE_RE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _clean_heading(raw: str) -> str:
    """'any\\_value [#](#link)' → 'any_value'"""
    raw = re.sub(r"\[#\]\([^)]*\)", "", raw)   # remove [#](…) anchors
    raw = raw.replace("\\_", "_")              # unescape underscores
    return raw.lstrip("#").strip()


def _is_noise_heading(heading: str) -> bool:
    _noise = {"back to top", "source code in", "previous", "next", "© copyright", "showing first"}
    h = heading.lower()
    return any(p in h for p in _noise)


def _extract_signature(block: str) -> str:
    """Return first fenced code block if it looks like a function signature."""
    m = re.search(r"```[^\n]*\n(.*?)```", block, re.DOTALL)
    if m:
        sig = m.group(1).strip()
        if len(sig) < 300 and "import" not in sig:
            return sig
    return ""


def _split_on_level(text: str, level: int) -> List[str]:
    prefix = "#" * level + " "
    return re.split(r"^" + re.escape(prefix), text, flags=re.MULTILINE)


def _split_if_oversized(full_text: str, heading: str, max_chars: int) -> List[str]:
    """Split a block that exceeds max_chars at paragraph boundaries."""
    if len(full_text) <= max_chars:
        return [full_text]
    parts, current, current_len = [], [], 0
    for para in full_text.split("\n\n"):
        if current_len + len(para) > max_chars and current:
            parts.append("\n\n".join(current))
            current = [f"## {heading} (continued)", para]
            current_len = sum(len(c) for c in current)
        else:
            current.append(para)
            current_len += len(para)
    if current:
        parts.append("\n\n".join(current))
    return parts or [full_text]


# ── Core chunker ──────────────────────────────────────────────────────────────

def chunk_markdown_file(markdown: str, filename: str) -> List[Dict]:
    """
    Chunk a single .md file into function-level chunks.

    Returns list of dicts:
        text        – full chunk text (heading + body)
        heading     – clean function/method name
        signature   – extracted function signature (if found)
        category    – inferred topic category
        source_file – original filename
        chunk_type  – 'function' | 'function_part_N' | 'full_file'
    """
    cleaned = _strip_noise(markdown)

    h2_parts = _split_on_level(cleaned, 2)
    h3_parts = _split_on_level(cleaned, 3)
    n_h2 = len(h2_parts) - 1   # number of ## headings
    n_h3 = len(h3_parts) - 1   # number of ### headings

    # ── Routing logic ───────────────────────────────────────────────────────
    # Priority 1: >= 3 h3 headings  →  class/method page  →  split on ###
    #   Why 3? Class pages always have many methods. A single ## function
    #   page can have 1-2 ### sub-sections (e.g. "### Example"), but never 3+
    #   at the function-boundary level.
    #
    # Priority 2: >= 1 h2 heading   →  split on ##
    #   Covers both single-function files (1 ##) and big reference pages (N ##)
    #
    # Priority 3: >= 1 h3 heading   →  split on ###
    #   Rare case: file has only ### without any ##
    #
    # Fallback: whole file as one chunk
    # ────────────────────────────────────────────────────────────────────────
    if n_h3 >= 3:
        raw_sections, prefix = h3_parts, "### "
    elif n_h2 >= 1:
        raw_sections, prefix = h2_parts, "## "
    elif n_h3 >= 1:
        raw_sections, prefix = h3_parts, "### "
    else:
        return _fallback_chunk(cleaned, filename)

    chunks: List[Dict] = []

    for raw in raw_sections[1:]:          # skip preamble before first heading
        lines       = raw.split("\n", 1)
        heading_raw = lines[0].strip()
        body        = lines[1].strip() if len(lines) > 1 else ""
        heading     = _clean_heading(heading_raw)

        if _is_noise_heading(heading):
            continue

        full_text = f"{prefix}{heading_raw}\n\n{body}".strip()

        if len(full_text) < MIN_CHUNK_CHARS:
            continue

        sub_blocks = _split_if_oversized(full_text, heading, MAX_CHUNK_CHARS)

        for i, block in enumerate(sub_blocks):
            chunk_type = "function" if len(sub_blocks) == 1 else f"function_part_{i+1}"
            chunks.append({
                "text":        block,
                "heading":     heading,
                "signature":   _extract_signature(block),
                "category":    _infer_category(filename, heading),
                "source_file": filename,
                "chunk_type":  chunk_type,
            })

    return chunks if chunks else _fallback_chunk(cleaned, filename)


def _fallback_chunk(text: str, filename: str) -> List[Dict]:
    """Whole file as one chunk when no usable headings found."""
    if len(text.strip()) < MIN_CHUNK_CHARS:
        return []
    m = re.search(r"^#{1,3} (.+)", text, re.MULTILINE)
    heading = _clean_heading(m.group(1)) if m else filename.replace(".md", "")
    return [{
        "text":        text,
        "heading":     heading,
        "signature":   _extract_signature(text),
        "category":    _infer_category(filename, heading),
        "source_file": filename,
        "chunk_type":  "full_file",
    }]


# ── File / directory loading ──────────────────────────────────────────────────

def load_markdown_file(path: Path) -> List[Dict]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")
    return chunk_markdown_file(text, path.name)


def load_all_documentation(md_path: str) -> List[Dict]:
    project_root = Path(__file__).resolve().parents[2]
    abs_path     = (project_root / md_path).resolve()

    print(f"Loading docs from: {abs_path}")
    md_files = sorted(abs_path.rglob("*.md"))

    if not md_files:
        print(f"⚠  No .md files found in {abs_path}")
        return []

    print(f"Found {len(md_files)} markdown files\n")
    all_chunks: List[Dict] = []

    for f in md_files:
        chunks = load_markdown_file(f)
        all_chunks.extend(chunks)
        print(f"  {f.name:<55} → {len(chunks):>3} chunk(s)")

    print(f"\nTotal chunks: {len(all_chunks)}\n")
    return all_chunks


# ── Chroma helpers ────────────────────────────────────────────────────────────

def setup_chroma_collection(
    db_path: str,
    collection_name: str,
    embedding_model_name: str,
) -> chromadb.Collection:

    script_dir  = Path(__file__).parent
    abs_db_path = (script_dir / db_path).resolve()
    abs_db_path.mkdir(parents=True, exist_ok=True)

    print(f"Setting up Chroma collection")
    print(f"  Path      : {abs_db_path}")
    print(f"  Collection: {collection_name}")
    print(f"  Embedder  : {embedding_model_name}\n")

    client = chromadb.PersistentClient(path=str(abs_db_path))
    # Must match the retrieval side: normalized embeddings + cosine space.
    # Without normalize_embeddings the docs land unnormalized while queries are
    # normalized, so similarity scores (and rankings) are meaningless.
    ef     = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model_name,
                normalize_embeddings=True)

    try:
        client.delete_collection(name=collection_name)
        print("  Existing collection deleted (fresh rebuild)")
    except Exception:
        print("  Creating new collection")

    collection = client.create_collection(
        name=collection_name,
        embedding_function=ef,
        metadata={
            "description": "Daft docs — one chunk per function/method",
            "hnsw:space": "cosine",
        },
    )
    print("  ✓ Collection ready\n")
    return collection


def ingest_chunks(
    chunks: List[Dict],
    collection: chromadb.Collection,
    batch_size: int = 100,
):
    print(f"Embedding & storing {len(chunks)} chunks …\n")

    ids = [f"chunk_{i:05d}" for i in range(len(chunks))]

    # mxbai retrieval is asymmetric: the query carries the instruction prefix,
    # documents are embedded as-is. Embedding the clean chunk text (rather than
    # a passage-prefixed string) also means retrieval can return exactly what
    # was embedded — no instruction noise leaking into codegen prompts.
    documents = [c["text"] for c in chunks]

    metadatas = [
        {
            # 2. Store the CLEAN text in metadata so we can display it nicely later
            "original_text": c["text"], 
            
            "heading":     c.get("heading",     ""),
            "signature":   c.get("signature",   ""),
            "category":    c.get("category",    "General"),
            "source_file": c.get("source_file", ""),
            "chunk_type":  c.get("chunk_type",  "unknown"),
        }
        for c in chunks
    ]

    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        collection.add(
            ids       = ids[start:end],
            documents = documents[start:end],
            metadatas = metadatas[start:end],
        )
        print(f"  Stored chunks {start}–{end-1}")

    print(f"\n✓ {len(chunks)} chunks stored\n")


def validate_collection(collection: chromadb.Collection, expected: int):
    actual = collection.count()
    status = "✓" if actual == expected else "⚠ MISMATCH"
    print(f"Validation: expected={expected}  actual={actual}  {status}\n")

    test_queries = [
        "group by and aggregate sum",
        "filter rows where column equals value",
        "user defined function udf",
        "date difference between two columns",
        "handle null values fill_null",
    ]
    print("Sample retrieval check:")
    for q in test_queries:
        results = collection.query(query_texts=[q], n_results=3)
        top = [
            f"{m.get('heading','?')} [{m.get('category','?')}]"
            for m in results["metadatas"][0]
        ]
        print(f"  Q: '{q}'")
        for i, t in enumerate(top, 1):
            print(f"     {i}. {t}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("DAFT DOCUMENTATION INGESTION — MARKDOWN (function-level chunks)")
    print("=" * 70 + "\n")

    chunks = load_all_documentation(DOCS_MD_PATH)
    if not chunks:
        print("❌ No chunks generated. Check DOCS_MD_PATH.")
        return

    collection = setup_chroma_collection(VECTOR_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL)
    ingest_chunks(chunks, collection)
    validate_collection(collection, len(chunks))

    print("=" * 70)
    print("✓ INGESTION COMPLETE")
    print("=" * 70)
    print(f"\nDB     : {Path(__file__).parent / VECTOR_DB_PATH}")
    print(f"Chunks : {len(chunks)}")
    print()


if __name__ == "__main__":
    main()