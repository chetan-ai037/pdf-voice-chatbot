from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
import numpy as np

# Lazy loading with caching
_embedder = None
_reranker = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def build_vector_store(text):
    # Optimized chunking for better accuracy: balanced chunk size with good overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Optimal size for context preservation
        chunk_overlap=100  # Increased overlap for better context continuity and accuracy
    )
    chunks = splitter.split_text(text)
    
    # Filter out empty or very short chunks - optimized list comprehension
    chunks = [chunk.strip() for chunk in chunks if chunk.strip() and len(chunk.strip()) > 30]

    if not chunks:
        raise ValueError("No valid text chunks found in the uploaded PDF. The PDF might be empty or contain only images.")

    embedder = get_embedder()
    # Batch processing with optimized settings for speed
    embeddings = embedder.encode(
        chunks, 
        normalize_embeddings=True, 
        show_progress_bar=False,
        batch_size=32,  # Larger batch for faster processing
        convert_to_numpy=True  # Faster than tensor
    )

    if len(embeddings.shape) == 1:
        embeddings = np.expand_dims(embeddings, axis=0)

    # Use IndexFlatIP (Inner Product) for cosine similarity with normalized vectors
    # This is equivalent to cosine similarity when vectors are normalized
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype('float32'))

    return index, chunks


def retrieve_secure(query, index, chunks, k=10, use_cache=False):
    """
    Optimized retrieval for accuracy: retrieve more chunks for better context coverage.
    k=10 provides better accuracy while maintaining reasonable speed.
    """
    embedder = get_embedder()
    # Single query encoding is already fast, no need for batch
    q_emb = embedder.encode(
        [query], 
        normalize_embeddings=True, 
        show_progress_bar=False,
        convert_to_numpy=True
    )
    q_emb = q_emb.astype('float32')
    
    # Optimized search - limit k for faster processing
    search_k = min(k, len(chunks))
    similarities, indices = index.search(q_emb, search_k)

    # Optimized filtering - use numpy operations where possible
    valid_mask = (indices[0] >= 0) & (indices[0] < len(chunks))
    valid_indices = indices[0][valid_mask]
    
    # Direct retrieval using valid indices
    retrieved = [chunks[int(idx)] for idx in valid_indices]
    
    # Optimized score calculation
    valid_similarities = similarities[0][valid_mask]
    similarities_clamped = np.clip(valid_similarities, -1, 1)
    scores = (1 - similarities_clamped).tolist()

    return retrieved, scores


def rerank_chunks(query, chunks, use_reranking=True):
    """Rerank chunks using cross-encoder. Can be disabled for speed."""
    if not use_reranking or len(chunks) <= 3:
        return chunks[:5]  # Return top 5 without reranking for speed
    
    reranker = get_reranker()
    pairs = [[query, c] for c in chunks]
    scores = reranker.predict(pairs, show_progress_bar=False)

    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:5]]