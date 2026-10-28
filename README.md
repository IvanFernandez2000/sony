# 🎬 Movie Intelligence Explorer

Explore 1,590 movies with natural-language semantic search, streaming availability across Latin America, and monthly consumption metrics.

**Live app:** https://movie-intelligence-explorer.streamlit.app/

> The app is hosted on Streamlit Community Cloud's free tier, which puts apps to sleep after 12 hours without traffic. If you see a "This app is asleep" page, click the button to wake it up; the first load takes a minute or two.

---

## Features

- **Semantic search in real time.** Describe what you want ("dark psychological thrillers about obsession") and movies are ranked by meaning, not keyword matching. Movie embeddings are precomputed; each query is embedded live.
- **Hybrid search.** An optional genre filter restricts candidates before semantic ranking.
- **Movie details.** Title, year, genres, plot, rating, votes, runtime, director and principal cast, with explicit fallbacks for missing values.
- **Streaming availability.** Countries, platforms and platform type (subscription or free with ads) as of the June 2026 snapshot.
- **Consumption performance.** Monthly streams and total minutes, filterable by country and platform, with an optional split into one line per country or platform.

---

## How to run locally

Requires Python 3.10–3.12.

```bash
git clone https://github.com/IvanFernandez2000/sony
cd sony

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

The app opens at http://localhost:8501. On first run, the embedding model (~90 MB) is downloaded from Hugging Face and cached.

Precomputed embeddings are included in `artifacts/`. To regenerate them (only needed if the movie data or the embedding text changes):

```bash
python build_embeddings.py
```

To run the data validations on their own:

```bash
python data_prep.py
```

---

## Project structure

```
├── app.py                  # Streamlit UI (layout, widgets, state)
├── data_prep.py            # Load, normalize and validate the three datasets
├── search.py               # Embedding text, model loading, semantic + hybrid search
├── metrics.py              # Consumption filtering, summary metrics, monthly series
├── build_embeddings.py     # Offline step: precompute movie embeddings
├── artifacts/
│   ├── movie_embeddings.npy   # 1,590 × 384 float32 matrix (~2.4 MB)
│   └── movie_ids.npy          # Movie order, used to check alignment on load
├── data/                   # dataset_A.csv, dataset_B.csv, dataset_C.csv
├── experiments/
│   └── 03_search_eval.ipynb   # Evaluation used to choose the embedding text
└── requirements.txt
```

---

## Architecture

**Flow.** There are two stages. Offline, `build_embeddings.py` converts every movie into a vector and saves the matrix to `artifacts/`. At runtime, the app loads the data and that matrix once, and for each search it only embeds the user's query and compares it with all movie vectors.

**Design principle: logic is separated from the interface.** `data_prep.py`, `search.py` and `metrics.py` contain no Streamlit code, so the exact same functions are used by the app, the evaluation notebook and the offline script. `app.py` only handles layout and user interaction.

**Streamlit execution model.** Streamlit re-runs the whole script on every interaction. Expensive steps are cached:

- `@st.cache_data` for the datasets and embeddings (returns a safe copy on each run).
- `@st.cache_resource` for the embedding model (one shared object, never copied).

User choices that must survive re-runs (the query and the selected movie) are stored in `st.session_state`.

**Why Streamlit.** It allowed building the whole app in Python, with no separate frontend or API to maintain, which kept the surface area small enough to understand and explain every part. The trade-off is less control over the visual design than a custom frontend.

---

## Data integration

### The three datasets have different grains

| Table | Source | Rows | Grain (validated) |
|---|---|---|---|
| `movies` | dataset_A | 1,590 | 1 row per movie |
| `availability` | dataset_B | 10,358 | 1 row per movie × country × platform (single snapshot: June 2026) |
| `consumption` | dataset_C | 62,022 | 1 row per movie × month × country × platform (Jan 2023 – Jun 2026) |

All three tables contain exactly the same 1,590 movies, keyed by IMDb ID.

### Key rule: availability and consumption are never joined together

Joining them on movie ID multiplies every consumption row by every availability row of the same movie. I measured it: the result grows from 62,022 to 432,195 rows, and total streams inflate from 9.0 million to 62.7 million, almost **7×**. Instead, each table is joined only with `movies` and shown in its own panel.

### Normalization

- **Column names.** The files use three naming styles (`TITLE_ID`, `IMDB_ID`, `imdb_id`), and they don't match the data dictionary, which refers to `TITLE_ID` everywhere. All columns are lowercased and the key is renamed to `title_id`.
- **Dropped columns.** `ampere_title` (dataset A's title is used as the main title; the two differ for 406 movies) and `month_year` (a single value in all rows; kept as the snapshot date shown in the UI).
- **Types.** `month` is parsed as a date for time-series charts.
- **Genres.** `"Comedy, Drama"` is split into a list for exact filtering. A substring search for "Music" would also match "Musical" (101 movies instead of the correct 68).

### Validations

`data_prep.py` checks, with assertions that stop the app if they fail:

- **Grain uniqueness** for each table (equivalent to a primary key).
- **Referential integrity:** every movie in availability and consumption exists in movies (equivalent to a foreign key).
- **Valid values:** no negative streams or minutes.
- `platform_type` depends only on `platform`, which is why it is not part of the availability key.

### Differences between sources (documented, not "fixed")

- **Availability is a current snapshot; consumption is historical.** 20% of the movie × country × platform combinations with consumption are not in today's availability. This is expected: a movie can have been streamed on a platform in 2024 and no longer be available there.
- **Coverage differs.** Availability covers 8 countries and 19 platforms. Consumption covers 4 countries (Argentina, Brazil, Colombia, Mexico) and 4 platforms.
- **Platform names follow each source.** Consumption reports "Amazon" as one platform; availability distinguishes "Amazon Prime Video" and "Amazon Other". Since the tables are never joined, names were not unified, and the UI states the difference.

### Missing data

- No rows are dropped. Missing director, cast or genres are shown as "Not available".
- The empty `original_platform` values (5,211 rows) mean "not an original", not missing data.
- **Months with no consumption records are shown as gaps in the charts, not as zero.** 857 of the 1,590 movies have missing months within their consumption period. Connecting the surrounding points would suggest data that doesn't exist, and filling with zero would assume something the dataset doesn't say.
- **Minutes per stream** is computed as total minutes ÷ total streams (ratio of sums), not as the average of per-row ratios, which would give a tiny month the same weight as a large one.

---

## Semantic search

### How it works

1. Each movie is represented by a text built from its genres and plot.
2. The text is converted into a 384-dimensional vector (embedding) by a sentence-transformer model. This is done once, offline.
3. At runtime, the user's query is converted into a vector with the same model.
4. Movies are ranked by **cosine similarity** between the query vector and each movie vector. All vectors are normalized to length 1, so cosine similarity reduces to a dot product: one matrix multiplication scores all movies at once.

### Model: `all-MiniLM-L6-v2`

- Small (~90 MB) and fast on CPU, which fits a free hosting tier with no GPU.
- Trained for sentence and short-paragraph similarity; the median plot is ~300 characters.
- Runs locally: no API key, no per-query cost, and queries are not sent to a third party.
- Trade-off: not the most accurate model available. Larger models (e.g. `all-mpnet-base-v2`) or API-based embeddings would likely improve quality at a higher operational cost.

### Choosing what to embed (evaluated, not assumed)

I compared three texts on the full dataset with the assignment's example queries plus my own, counting how many of the top 5 results were actually relevant (checked against each movie's genres and plot):

| Query | Plot only | Genres + plot | Title + genres + plot |
|---|---|---|---|
| dark psychological thrillers about obsession | 4/5 | **5/5** | 2/5 |
| family movies about overcoming loss | 0/5 | 0/5 | 0/5 |
| animated adventures | 3/5 | 3/5 | **4/5** |
| movies about artificial intelligence | 4/5 | **5/5** | 4/5 |
| something to watch with kids on a rainy day | 1/5 | **2/5** | 0/5 |
| survival story in space | 3/5 | **4/5** | **4/5** |

**Chosen: genres + plot.** Titles added literal-word noise: for "dark psychological thrillers about obsession", the title variant returned *Lover Stalker Killer*, *Hypnotic* and *Control Freak* because of words in their titles, and even *Orion and the Dark*, an animated family movie, because its title contains "Dark". For "a lonely hitman losing his nerve", it surfaced *HIT: The 3rd Case*, an Indian police thriller, because of the word "hit". Genres, on the other hand, improved relevance for genre-oriented queries.

The full comparison is in `experiments/03_search_eval.ipynb`. A larger, labeled set of test queries would make this evaluation more robust; with six queries it is indicative, not conclusive.

### Hybrid search: why the genre filter exists

No variant returned a single Family movie for "family movies about overcoming loss", even though the dataset has 127. The model reads "family" as "a family appears in the plot", not "suitable for families", and many horror films and dramas are about families. This is a limitation of purely semantic search, not of a particular text variant.

The fix is a structured filter: select "Family" and search "overcoming loss". The filter guarantees the genre; the embedding ranks by theme.

### Other decisions

- **No vector database.** 1,590 × 384 float32 values take ~2.4 MB, and scoring all movies is a single matrix multiplication that takes milliseconds. A vector database would add infrastructure without benefit at this scale.
- **Alignment check.** Row *i* of the embedding matrix must belong to movie *i*. The movie order is saved next to the embeddings and checked on load; if the data order changed, the app fails with a clear message instead of silently returning the wrong movies.
- **No similarity threshold.** Scores are only meaningful relative to each other within a query (the best match scored 0.29 for one correct query and 0.51 for another), so a fixed cutoff would discard good results. Instead, the user chooses how many results to see (5–30).
- **Field order in the text.** The model truncates inputs at ~256 tokens, so genres go before the plot: if a long plot is cut, the genres are kept.

---

## User experience

- **Two-column layout:** results on the left, details on the right, so you can browse results without losing the search.
- **The best match is selected automatically** after each search.
- **Example queries** (from the assignment) as one-click buttons.
- **Explicit states:** a welcome message with no query, a spinner while loading, a warning with a suggested action when filters return no results, and messages when a movie has no availability or no consumption for the selected filters.
- Notes under the charts explain gaps and coverage differences between sources.

---

## Deployment

Deployed on **Streamlit Community Cloud** from this GitHub repository.

- **Python 3.12.** Community Cloud defaulted to Python 3.14, for which the pinned NumPy version has no prebuilt wheel. NumPy was compiled from source, the install took 46 minutes, and the app then failed at startup. Redeploying on Python 3.12 fixed it. I changed only the Python version, keeping the dependencies that had already resolved correctly, to isolate the cause.
- **Pinned dependency versions** in `requirements.txt`, so the server installs exactly what was tested.
- **Precomputed embeddings are committed** (`artifacts/`, ~2.5 MB), so the deployed app never re-embeds the catalog. `build_embeddings.py` makes them reproducible.
- **Known inefficiency:** on Linux, the default PyTorch package includes several GB of CUDA libraries that are useless on a CPU-only server. A CPU-only build would make installs much lighter. I kept the default build because it works and changing the package index adds dependency-resolution risk.

---

## Limitations

- **English queries work best.** The model was trained on English. A Spanish query ("películas de terror en una casa embrujada") partly works but matches literal words, such as Spanish titles containing "películas". A multilingual model would address this.
- **Ambiguous intent.** As shown above, "family" or "kids" are read as plot content, not audience. The genre filter mitigates this, but the user has to use it.
- **Hub movies.** Some movies (e.g. *On Fire*) appear in results for several unrelated queries, a known effect in vector search.
- **Truncation.** A few very long plots exceed the model's input limit and their endings are ignored.
- **Data quality, kept as-is and documented:**
  - 1,254 consumption rows have 0 streams; 75 have minutes but no streams, and 414 have streams but no minutes.
  - 3 movies have consumption before their release year (*Furiosa*, *Frankenstein*, *Cortafuego*), possibly from premieres or festivals.
  - Ratings are whole numbers, possibly rounded at the source.
  - Some titles are shared by different movies: for example, the dataset has two different movies called *The Killer* and two called *Ballerina*. To tell them apart, results always show the release year next to the title, and the app identifies movies by their IMDb ID, never by title.
- **Testing.** Data correctness is enforced with runtime assertions; there is no automated unit test suite.
- **Cold start.** The free tier sleeps after inactivity, and waking up includes loading the model.

---

## Possible improvements

- Multilingual embedding model for Spanish and Portuguese queries.
- A larger labeled query set to evaluate search quality more rigorously.
- Combine semantic search with keyword search (e.g. BM25) for names and exact terms.
- More structured filters: year, minimum rating, availability in a given country.
- Unit tests for `metrics.py` and `search.py`.
- CPU-only PyTorch build for lighter deployments.

---

## AI tools used


I used **Claude** throughout the project as a technical mentor and pair programmer:

- **Learning:** explanations of embeddings, cosine similarity, normalization, Streamlit's execution model and deployment concepts.
- **Code:** drafts of the modules, which I reviewed, ran and modified, and whose design I discussed before adopting.
- **Debugging:** interpreting errors (import issues, unsaved files, Git staging, the deployment's Python version) and choosing fixes.

What I did and verified myself:

- Ran the embedding experiments and the comparison of text variants, and chose the variant based on the results.
- Tested the app as a user and decided on adjustments.
- Set up the repository and the deployment, and resolved the deployment issues.
- Can explain every technical decision in this README and modify the code.