"""
rag_engine.py — RAG Engine with Embeddings, Caching, and Similarity Search

Chunks transcripts, generates embeddings using sentence-transformers,
caches them for fast startup, and provides cosine-similarity-based retrieval.
"""

import os
import pickle
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

# Load .env file variables, including GOOGLE_API_KEY
load_dotenv()
if "GOOGLE_API_KEY" in os.environ:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

from transcript_extractor import fetch_transcripts, VIDEOS

# ─── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 200       # words per chunk
CHUNK_OVERLAP = 50     # overlapping words between chunks
TOP_K = 5              # number of chunks to retrieve

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "rag_cache.pkl")


# ─── Chunking ──────────────────────────────────────────────────────────────────

def chunk_transcripts(transcripts: dict) -> list[dict]:
    """
    Split transcripts into overlapping chunks for embedding.

    Each chunk contains ~CHUNK_SIZE words with CHUNK_OVERLAP word overlap.
    Uses sentence boundaries (`.` `?` `!`) to avoid splitting sentences mid-way.
    Preserves video metadata and approximate timestamps.
    
    Returns:
        list[dict]: Each chunk has keys: text, video_id, title, url, start_timestamp
    """
    chunks = []

    for video_id, data in transcripts.items():
        segments = data["segments"]
        if not segments:
            continue

        # Build word-level index with timestamp mapping
        words = []
        word_timestamps = []
        for seg in segments:
            seg_words = seg["text"].split()
            for w in seg_words:
                words.append(w)
                word_timestamps.append(seg["timestamp"])

        full_text = " ".join(words)
        # Split into sentences using regex on punctuation
        sentences = re.split(r'(?<=[.!?])\s+', full_text)
        sentences = [s.strip() for s in sentences if s.strip()]

        sentence_data = [] # (sentence, word_count, timestamp)
        current_word_idx = 0
        for s in sentences:
            s_words = len(s.split())
            if current_word_idx < len(word_timestamps):
                ts = word_timestamps[current_word_idx]
            else:
                ts = word_timestamps[-1] if word_timestamps else "00:00"
            sentence_data.append((s, s_words, ts))
            current_word_idx += s_words

        # Build chunks from sentences
        i = 0
        while i < len(sentence_data):
            start_i = i
            chunk_sentences = []
            chunk_word_count = 0
            start_ts = sentence_data[i][2]

            # Fill chunk up to CHUNK_SIZE target
            while i < len(sentence_data) and chunk_word_count < CHUNK_SIZE:
                chunk_sentences.append(sentence_data[i][0])
                chunk_word_count += sentence_data[i][1]
                i += 1

            chunks.append({
                "text": " ".join(chunk_sentences),
                "video_id": video_id,
                "title": data["title"],
                "url": data["url"],
                "start_timestamp": start_ts,
            })

            # Overlap: backtrack by roughly CHUNK_OVERLAP words
            # Guaranteed forward progress by never backtracking to start_i
            if i < len(sentence_data):
                overlap = 0
                while i > start_i + 1 and overlap < CHUNK_OVERLAP:
                    i -= 1
                    overlap += sentence_data[i][1]

    return chunks


# ─── RAG Engine Class ──────────────────────────────────────────────────────────

class RAGEngine:
    """
    Retrieval-Augmented Generation engine.
    
    - Loads transcripts and chunks them
    - Generates embeddings with sentence-transformers
    - Caches embeddings to disk for fast reload
    - Provides cosine-similarity-based retrieval
    """

    def __init__(self, force_rebuild: bool = False):
        """Initialize the RAG engine, loading from cache if available."""
        print("🔧 Initializing RAG Engine...")

        self.model = None  # Lazy-load the model
        self.chunks = []
        self.embeddings = None
        self.transcripts = {}

        # Try loading from cache first
        if not force_rebuild and self._load_cache():
            print("  ✓ Loaded from cache")
            # Still need transcripts for display
            self.transcripts = fetch_transcripts()
        else:
            self._build_index()

    def _get_model(self) -> SentenceTransformer:
        """Lazy-load the sentence transformer model."""
        if self.model is None:
            print("  ⏳ Loading embedding model...")
            self.model = SentenceTransformer(MODEL_NAME)
            print(f"  ✓ Model loaded: {MODEL_NAME}")
        return self.model

    def _build_index(self):
        """Fetch transcripts, chunk them, generate embeddings, and cache."""
        print("  ⏳ Building index from scratch...")

        # Step 1: Fetch transcripts
        self.transcripts = fetch_transcripts()

        if not self.transcripts:
            print("  ⚠ No transcripts available!")
            return

        # Step 2: Chunk transcripts
        self.chunks = chunk_transcripts(self.transcripts)
        print(f"  ✓ Created {len(self.chunks)} chunks")

        # Step 3: Generate embeddings
        model = self._get_model()
        texts = [c["text"] for c in self.chunks]
        print(f"  ⏳ Generating embeddings for {len(texts)} chunks...")
        self.embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        print(f"  ✓ Embeddings shape: {self.embeddings.shape}")

        # Step 4: Cache to disk
        self._save_cache()

    def _save_cache(self):
        """Save chunks and embeddings to disk."""
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_data = {
            "chunks": self.chunks,
            "embeddings": self.embeddings,
        }
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(cache_data, f)
        print(f"  ✓ Cache saved to {CACHE_FILE}")

    def _load_cache(self) -> bool:
        """Load chunks and embeddings from disk cache."""
        if not os.path.exists(CACHE_FILE):
            return False
        try:
            with open(CACHE_FILE, "rb") as f:
                cache_data = pickle.load(f)
            self.chunks = cache_data["chunks"]
            self.embeddings = cache_data["embeddings"]
            print(f"  ✓ Cache loaded: {len(self.chunks)} chunks, embeddings {self.embeddings.shape}")
            return True
        except Exception as e:
            print(f"  ⚠ Cache load failed: {e}")
            return False

    def query(self, question: str, top_k: int = TOP_K) -> dict:
        """
        Perform a RAG query: request chunks and summarize with Gemini API.

        Args:
            question: The user's question
            top_k: Number of chunks to retrieve

        Returns:
            dict with keys: answer, contexts, sources
        """
        if not self.chunks or self.embeddings is None:
            return {
                "answer": "⚠ No data available. Please ensure transcripts are loaded.",
                "contexts": [],
                "sources": [],
            }

        # Encode the query
        model = self._get_model()
        query_embedding = model.encode([question], convert_to_numpy=True)

        # Compute cosine similarity
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Build results
        contexts = []
        sources = []
        context_text_for_prompt = ""
        for i, idx in enumerate(top_indices, 1):
            chunk = self.chunks[idx]
            score = float(similarities[idx])
            contexts.append({
                "text": chunk["text"],
                "score": round(score, 4),
                "video_title": chunk["title"],
                "video_url": chunk["url"],
                "timestamp": chunk["start_timestamp"],
            })
            source_str = f"{chunk['title']} @ {chunk['start_timestamp']}"
            if source_str not in sources:
                sources.append(source_str)
                
            context_text_for_prompt += f"--- Context {i} ---\nSource: {source_str}\nText: {chunk['text']}\n\n"

        # Generate answer using Gemini API (if configured)
        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            try:
                llm = genai.GenerativeModel('gemini-2.5-flash')
                prompt = (
                    "You are an expert AI assistant answering questions about deep learning and neural networks.\n"
                    "You MUST answer the question strictly based ONLY on the provided video transcript contexts below.\n"
                    "Do NOT hallucinate or use outside knowledge. If the answer is not contained in the context, "
                    "clearly state that you cannot answer based on the provided videos.\n\n"
                    f"Contexts:\n{context_text_for_prompt}\n"
                    f"Question: {question}\n\n"
                    "Answer:"
                )
                response = llm.generate_content(prompt)
                answer = response.text
                
                # Identify if Gemini failed to answer
                if "cannot answer" not in answer.lower():
                    answer += "\n\n*(✨ Generated by Gemini based on retrieved context)*"
                    
            except Exception as e:
                answer = f"⚠ Gemini API Error: {str(e)}\n\n"
                answer += "(Falling back to raw context snippets)\n\n"
                for ctx in contexts[:3]:
                    answer += f"> {ctx['text'][:400]}...\n\n"
        else:
            # Fallback if no .env config
            answer = "⚠ No `GOOGLE_API_KEY` found in environment or `.env` file. Unable to use Gemini for synthesis.\n\n"
            answer += "(Falling back to raw context snippets)\n\n"
            for ctx in contexts[:3]:
                answer += f"> {ctx['text'][:400]}...\n\n"

        return {
            "answer": answer,
            "contexts": contexts,
            "sources": sources,
        }

    def get_stats(self) -> dict:
        """Return statistics about the RAG index."""
        return {
            "total_chunks": len(self.chunks),
            "embedding_dim": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "total_videos": len(set(c["video_id"] for c in self.chunks)),
            "model": MODEL_NAME,
            "cache_exists": os.path.exists(CACHE_FILE),
        }


# ─── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🔍 Testing RAG Engine...\n")
    engine = RAGEngine()

    stats = engine.get_stats()
    print(f"\n📊 Index Stats: {stats}\n")

    test_query = "What is a neural network and how does it learn?"
    print(f"❓ Query: {test_query}\n")
    result = engine.query(test_query)
    print(result["answer"])
    print(f"\n📍 Sources: {result['sources']}")
