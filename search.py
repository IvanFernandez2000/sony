"""
Semantic search over movies using sentence embeddings.

How it works:
1. Each movie is turned into a text (genres + plot) and then into a
   384-dimensional vector (embedding). This is done once, ahead of time,
   by build_embeddings.py, and saved to artifacts/.
2. At runtime, the user's query is turned into a vector with the same model.
3. Movies are ranked by cosine similarity between the query vector and each
   movie vector. Since all vectors are normalized (length 1), cosine
   similarity is just a dot product.
4. Optionally, a genre filter restricts the candidates before ranking
   (hybrid search: structured filter + semantic similarity).
"""
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

TEXT_VARIANTS = ["plot", "genres_plot", "title_genres_plot"]
# Chosen after evaluation (see experiments/03_search_eval.ipynb):
# titles added literal-word noise; genres improved relevance.
EMBEDDING_VARIANT = "genres_plot"

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
EMBEDDINGS_PATH = ARTIFACTS_DIR / "movie_embeddings.npy"
IDS_PATH = ARTIFACTS_DIR / "movie_ids.npy"


def build_embedding_text(movies, variant=EMBEDDING_VARIANT):
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


def load_embeddings(movies):
    """Load precomputed embeddings and check they are aligned with movies.

    Row i of the matrix must belong to movie i. If the CSV order changed
    since the embeddings were built, results would silently point to the
    wrong movies, so we fail loudly instead.
    """
    embeddings = np.load(EMBEDDINGS_PATH)
    ids = np.load(IDS_PATH)
    if not np.array_equal(ids, movies["title_id"].to_numpy(dtype=str)):
        raise ValueError(
            "Embeddings are not aligned with the movies table. "
            "Run `python build_embeddings.py` again."
        )
    return embeddings


def search(query, model, embeddings, movies, top_k=10, genres=None):
    """Return the top_k movies most similar to the query, with a 'score' column.

    genres: optional list of genres. If given, only movies that have at least
    one of them are considered (OR logic, like most shopping-site filters).
    """
    query = query.strip()
    if not query:
        # Empty query: return an empty result with the same columns.
        return movies.head(0).assign(score=[])

    # 1) Candidates: all movies, or only those matching the genre filter.
    if genres:
        selected = set(genres)
        mask = movies["genre_list"].apply(lambda gs: bool(selected.intersection(gs)))
        candidate_idx = np.where(mask.to_numpy())[0]
    else:
        candidate_idx = np.arange(len(movies))

    if len(candidate_idx) == 0:
        return movies.head(0).assign(score=[])

    # 2) Rank candidates by similarity to the query.
    query_vector = model.encode(query, normalize_embeddings=True)   # shape (384,)
    scores = embeddings[candidate_idx] @ query_vector                # one score per candidate
    order = np.argsort(-scores)[:top_k]                              # best first

    results = movies.iloc[candidate_idx[order]].copy()
    results["score"] = scores[order]
    return results.reset_index(drop=True)