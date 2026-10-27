"""
Ejercicio 1: qué es un embedding y cómo se ordena por similitud.

Objetivo: ver con datos reales que el modelo convierte texto en vectores
y que "parecido en significado" se traduce en "vectores que apuntan
en direcciones parecidas".
"""
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# 1) Cargamos seis películas reales del dataset A, elegidas por ID.
IDS = [
    "tt14205554",  # K-Pop Demon Hunters
    "tt12042730",  # Project Hail Mary
    "tt1136617",   # The Killer (2023, Fincher)
    "tt22022452",  # Inside Out 2
    "tt26342662",  # M3GAN 2.0
    "tt12747748",  # Leave the World Behind
]
df = pd.read_csv("data/dataset_A.csv")
movies = df[df["TITLE_ID"].isin(IDS)].reset_index(drop=True)

# 2) Cargamos el modelo. La primera vez lo descarga (~90 MB) y queda en caché.
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3) Convertimos cada sinopsis en un vector.
#    normalize_embeddings=True deja cada vector con longitud 1.
#    Así, la similitud coseno se reduce a un simple producto punto.
movie_vectors = model.encode(
    movies["PLOT_SUMMARY"].tolist(), normalize_embeddings=True
)
print("Forma de la matriz:", movie_vectors.shape)  # (6, 384)
print("Primeros 5 números del primer vector:", movie_vectors[0][:5].round(3))
print("Longitud del primer vector:", np.linalg.norm(movie_vectors[0]).round(3))

# 4) Consultas del usuario: se procesan en el momento, igual que en la app.
queries = [
    "a lonely hitman losing his nerve",
    "family movies about growing up and emotions",
    "movies about artificial intelligence",
    "a mysterious catastrophe during a vacation",
]

for q in queries:
    q_vector = model.encode(q, normalize_embeddings=True)  # (384,)
    # Producto punto de la consulta contra las 6 películas a la vez -> 6 scores
    scores = movie_vectors @ q_vector
    order = np.argsort(-scores)  # índices de mayor a menor score

    print(f"\nConsulta: {q!r}")
    for i in order:
        print(f"  {scores[i]:.3f}  {movies.loc[i, 'ORIGINAL_TITLE']}")