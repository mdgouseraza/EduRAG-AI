from typing import List, Dict
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config.settings import EMBED_MODEL, TOP_K_RETRIEVAL
from rag.ingestion import get_embed_model


def retrieve(query: str, store: Dict, top_k: int = TOP_K_RETRIEVAL) -> List[Dict]:
    """Semantic retrieval: returns top_k most relevant chunks."""
    if not store or not store.get("chunks"):
        return []

    model = get_embed_model()
    query_vec = model.encode([query], show_progress_bar=False)
    query_vec = np.array(query_vec).astype("float32")
    faiss.normalize_L2(query_vec)

    index = store["index"]
    k = min(top_k, len(store["chunks"]))
    scores, indices = index.search(query_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = store["chunks"][idx]
        results.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "page": chunk["page"],
            "score": float(score)
        })

    return results


def format_context(results: List[Dict]) -> str:
    """Format retrieved chunks into a context string for the LLM."""
    if not results:
        return ""
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"[Source {i}: {r['source']}, Page {r['page']}]\n{r['text']}"
        )
    return "\n\n---\n\n".join(parts)


def format_sources(results: List[Dict]) -> str:
    """Format source citations for display."""
    if not results:
        return ""
    seen = set()
    lines = []
    for r in results:
        key = (r["source"], r["page"])
        if key not in seen:
            seen.add(key)
            lines.append(f"📄 **{r['source']}** — Page {r['page']}")
    return "\n".join(lines)
