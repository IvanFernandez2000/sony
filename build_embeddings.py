"""
Precompute movie embeddings and save them to artifacts/.

Run once (and again whenever the movie data or the embedding text changes):
    python build_embeddings.py

The app only loads these files; it never re-embeds the 1,590 movies.
Only the user's query is embedded at runtime.
"""
import numpy as np

from data_prep import load_data
from search import (
    ARTIFACTS_DIR,
    EMBEDDING_VARIANT,
    EMBEDDINGS_PATH,
    IDS_PATH,
    build_embedding_text,
    embed_texts,
    load_model,
)


def main():
    movies, _, _, _ = load_data()
    model = load_model()

    texts = build_embedding_text(movies, EMBEDDING_VARIANT)
    embeddings = embed_texts(model, texts).astype(np.float32)

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    # Save the movie order too, so the app can check alignment on load.
    np.save(IDS_PATH, movies["title_id"].to_numpy(dtype=str))

    print(f"Saved {embeddings.shape} embeddings ({EMBEDDING_VARIANT}) to {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()