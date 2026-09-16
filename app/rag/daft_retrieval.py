"""
RAG Retrieval for Daft Documentation
Retrieves top-k relevant chunks per pseudocode line with deduplication.
"""

from pathlib import Path
from typing import Dict, List, Set, Optional
import chromadb
from chromadb.utils import embedding_functions
import re


# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
VECTOR_DB_PATH = SCRIPT_DIR / "daft_embeddings"
COLLECTION_NAME = "daft_documentation"
EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
DEFAULT_TOP_K = 3  # Retrieve top-3 per line # can be changed too 


def load_chroma_collection(
    db_path: str = VECTOR_DB_PATH,
    collection_name: str = COLLECTION_NAME,
    embedding_model: str = EMBEDDING_MODEL
) -> chromadb.Collection:
    """
    Load existing Chroma collection for querying.
    """
    abs_db_path = Path(db_path)

    # Check the store actually exists on disk. PersistentClient would otherwise
    # silently (re)create an empty DB, and retrieval would return nothing with
    # no error.
    if not (abs_db_path / "chroma.sqlite3").exists():
        raise FileNotFoundError(
            f"Chroma database not found at: {abs_db_path}\n"
            f"Please run the ingestion script first."
        )

    print(f"Loading Chroma collection from: {abs_db_path}")
    print(f"Loading Embedding Model: {embedding_model} (this may take a moment)...")
    
    client = chromadb.PersistentClient(path=str(abs_db_path))
    
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_model,
        normalize_embeddings=True
    )
    
    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=sentence_transformer_ef
        )
        print(f"✓ Collection '{collection_name}' loaded")
        print(f"  Total chunks: {collection.count()}")
        print()
        return collection
        
    except Exception as e:
        raise ValueError(
            f"Collection '{collection_name}' not found.\n"
            f"Please run the ingestion script first.\n"
            f"Error: {e}"
        )
    
def _extract_keywords(text: str) -> str:
    """
    Internal Helper: Distills pseudocode into high-value search terms.
    Heavily anchors explicit function hints to overcome stopword bias.
    """
    keywords = []
    
    # 1. Extract exact function names from the [Function: ...] block
    # Matches words right before () inside the brackets
    func_block = re.search(r"\[Function:\s*(.*?)\]", text)
    if func_block:
        funcs = re.findall(r"([a-zA-Z_]+)\(\)", func_block.group(1))
        for f in funcs:
            # Anchor it strongly: "daft when function"
            keywords.append(f"daft {f} function")
            
    # 2. Catch text in quotes (column names)
    quotes = re.findall(r"(['\"].*?['\"])", text)
    keywords.extend(quotes)
    
    return " ".join(keywords)


def _create_search_payload(lines: List[str]) -> List[str]:
    """
    Internal Helper: Formats queries for the mxbai model.
    It combines the original line with extracted keywords and the required instruction.
    """
    payloads = []
    for line in lines:
        keywords = _extract_keywords(line)
        # mxbai models perform best with this specific instruction prefix
        instruction = f"Represent this sentence for searching relevant passages: {line} {keywords}"
        payloads.append(instruction)
    return payloads


def deduplicate_chunks(raw_docs: List[str]) -> List[str]:
    """
    Remove duplicate chunks while preserving the order of relevance.
    """
    seen = set()
    unique_chunks = []
    
    for doc in raw_docs:
        # Normalize: strip whitespace to ensure perfect string matching
        clean_doc = doc.strip()
        
        if clean_doc and clean_doc not in seen:
            seen.add(clean_doc)
            unique_chunks.append(clean_doc)
            
    return unique_chunks

def retrieve_docs(
    pseudo_code_lines: List[str],
    collection: chromadb.Collection = None,
    top_k: int = DEFAULT_TOP_K
) -> List[str]:
    """
    Retrieve relevant documentation for all pseudocode lines.
    Main function for RAG retrieval.
    """
    # Load collection if not provided
    if collection is None:
        collection = load_chroma_collection()
    
    print(f"Processing {len(pseudo_code_lines)} lines of pseudocode...")
    
    # 1. Prepare Batch Queries
    search_queries = _create_search_payload(pseudo_code_lines)
    
    try:
        print("Querying vector database (Batch Mode)...")
        
        # 2. Execute Batch Query
        # We send ALL lines at once. Chroma handles the parallelization.
        results = collection.query(
            query_texts=search_queries,
            n_results=top_k
        )
        
        # 3. Flatten Results
        # Prefer the clean text stored in metadata['original_text']; the embedded
        # document may carry an instruction prefix we must not inject into the
        # codegen prompt. Fall back to the document only if metadata is absent.
        all_retrieved_docs = []
        metadatas = results.get('metadatas') or []
        documents = results.get('documents') or []
        for i, meta_list in enumerate(metadatas):
            doc_list = documents[i] if i < len(documents) else []
            for j, meta in enumerate(meta_list or []):
                text = (meta or {}).get('original_text') or (doc_list[j] if j < len(doc_list) else "")
                if text:
                    all_retrieved_docs.append(text)

        # 4. Deduplicate
        # We use the separate function to keep logic clean, just like your original structure
        unique_chunks = deduplicate_chunks(all_retrieved_docs)
        
        print(f"✓ Retrieval complete. Found {len(unique_chunks)} unique relevant docs.")
        return unique_chunks

    except Exception as e:
        print(f"⚠ Error during batch retrieval: {e}")
        return []
    
def format_chunks_for_prompt(unique_chunks: List[str], max_chunks: int = None) -> str:
    """
    Format retrieved chunks into a string for inclusion in code generation prompt.
    """
    if max_chunks:
        chunks_to_use = unique_chunks[:max_chunks]
    else:
        chunks_to_use = unique_chunks
    
    formatted_parts = []
    formatted_parts.append("=== RETRIEVED DAFT DOCUMENTATION ===\n")
    
    for idx, chunk in enumerate(chunks_to_use, 1):
        formatted_parts.append(f"--- Document {idx} ---")
        formatted_parts.append(f"\n{chunk}\n")
    
    formatted_parts.append("=== END DOCUMENTATION ===\n")
    
    return '\n'.join(formatted_parts)


# Example usage
if __name__ == "__main__":
    # Test configuration
    test_pseudo_code_lines = [
        "Group by blood_group and count patients",
        "Filter rows where age > 18",
        "create a new column that groups patient data by blood group and calculate the count of patients in each blood group",
        "Create new column with title case names",
    ]
    
    try:
        # Retrieve docs
        results = retrieve_docs(test_pseudo_code_lines, top_k=3)
        pretty = format_chunks_for_prompt(results)

        print("=== Formatted Retrieved Documentation ===")
        print(pretty)           
        
    except Exception as e:
        print(f"❌ Error: {e}")