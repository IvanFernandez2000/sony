"""
Semantic search over movies using sentence embeddings.

How it works:
1. Each movie is turned into a text (e.g. genres + plot) and then into a
   384-dimensional vector (embedding). This is done once, ahead of time.
2. At runtime, the user's query is turned into a vector with the same model.
3. Movies are ranked by cosine similarity between the query vector and each
   movie vector. Since all vectors are normalized (length 1), cosine
   similarity is just a dot product.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

TEXT_VARIANTS = ["plot", "genres_plot", "title_genres_plot"]


def build_embedding_text(movies, variant="genres_plot"):
    """Build the text that represents each movie for embedding.

    Short, high-value fields go first: the model truncates long inputs
    (~256 tokens), so the end of very long plots may be cut off.
    """
    plot = movies["plot_summary"].fillna("")
    genres = movies["genres"].fillna("")

    if variant == "plot":
        return plot
    if variant == "genres_plot":
        return "Genres: " + genres + ". Plot: " + plot
    if variant == "title_genres_plot":
        return "Title: " + movies["original_title"] + ". Genres: " + genres + ". Plot: " + plot
    raise ValueError(f"Unknown variant: {variant}. Options: {TEXT_VARIANTS}")


def load_model():
    """Load the embedding model (downloaded from Hugging Face on first use)."""
    return SentenceTransformer(MODEL_NAME)


def embed_texts(model, texts):
    """Turn a list of texts into a matrix of normalized vectors (n_texts × 384)."""
    return model.encode(
        list(texts),
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=True,
    )


def search(query, model, embeddings, movies, top_k=10):
    """Return the top_k movies most similar to the query, with a 'score' column.

    embeddings must be aligned with movies: row i of embeddings is movie i.
    """
    query = query.strip()
    if not query:
        # Empty query: return an empty result with the same columns.
        return movies.head(0).assign(score=[])

    query_vector = model.encode(query, normalize_embeddings=True)  # shape (384,)
    scores = embeddings @ query_vector                              # shape (n_movies,)
    top_idx = np.argsort(-scores)[:top_k]                           # best first

    results = movies.iloc[top_idx].copy()
    results["score"] = scores[top_idx]
    return results.reset_index(drop=True)