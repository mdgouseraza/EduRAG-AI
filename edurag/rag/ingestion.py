import os
import pickle
from typing import List, Dict
import numpy as np

try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader

from sentence_transformers import SentenceTransformer
import faiss

from config.settings import EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, VECTOR_STORE_DIR


_embed_model = None


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    """Extract text with page numbers from a PDF."""
    chunks_raw = []
    try:
        reader = PdfReader(pdf_path)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                chunks_raw.append({"text": text, "page": page_num})
    except Exception as e:
        print(f"PDF extraction error: {e}")
    return chunks_raw


def chunk_text(pages: List[Dict], doc_name: str) -> List[Dict]:
    """Split page texts into overlapping chunks."""
    chunks = []
    for page_data in pages:
        text = page_data["text"]
        page_num = page_data["page"]
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk = text[start:end]
            if chunk.strip():
                chunks.append({
                    "text": chunk,
                    "page": page_num,
                    "source": doc_name
                })
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def build_faiss_index(chunks: List[Dict]) -> tuple:
    """Build FAISS index from chunks."""
    model = get_embed_model()
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False, batch_size=32)
    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, embeddings


def ingest_pdf(pdf_path: str, doc_name: str, existing_store: Dict = None) -> Dict:
    """
    Full ingestion pipeline: extract → chunk → embed → index.
    Returns a store dict with index, chunks, and metadata.
    """
    pages = extract_text_from_pdf(pdf_path)
    if not pages:
        raise ValueError(f"No text found in {doc_name}. The PDF may be scanned/image-only.")

    new_chunks = chunk_text(pages, doc_name)
    if not new_chunks:
        raise ValueError(f"Could not create chunks from {doc_name}.")

    if existing_store and existing_store.get("chunks"):
        all_chunks = existing_store["chunks"] + new_chunks
    else:
        all_chunks = new_chunks

    index, embeddings = build_faiss_index(all_chunks)

    store = {
        "chunks": all_chunks,
        "index": index,
        "embeddings": embeddings,
        "documents": list({c["source"] for c in all_chunks})
    }

    save_store(store)
    return store


def save_store(store: Dict):
    """Persist the vector store to disk."""
    index_path = os.path.join(VECTOR_STORE_DIR, "index.faiss")
    meta_path = os.path.join(VECTOR_STORE_DIR, "metadata.pkl")
    faiss.write_index(store["index"], index_path)
    with open(meta_path, "wb") as f:
        pickle.dump({
            "chunks": store["chunks"],
            "documents": store["documents"]
        }, f)


def load_store() -> Dict:
    """Load the vector store from disk."""
    index_path = os.path.join(VECTOR_STORE_DIR, "index.faiss")
    meta_path = os.path.join(VECTOR_STORE_DIR, "metadata.pkl")
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        return None
    try:
        index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        return {
            "index": index,
            "chunks": meta["chunks"],
            "documents": meta["documents"]
        }
    except Exception:
        return None
