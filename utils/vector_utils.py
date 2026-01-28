from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
import numpy as np
embedder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
def build_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80
    )
    chunks = splitter.split_text(text)
    embeddings = embedder.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, chunks
def retrieve_secure(query, index, chunks, k=6):
    q_emb = embedder.encode([query])
    distances, indices = index.search(q_emb, k)
    retrieved = [chunks[i] for i in indices[0]]
    scores = distances[0].tolist()
    return retrieved, scores
def rerank_chunks(query, chunks):
    pairs = [[query, c] for c in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:3]]