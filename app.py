"""
Movie Intelligence Explorer: Streamlit app.

Run locally:
    streamlit run app.py

Important Streamlit concept: the whole script re-runs from top to bottom
on every user interaction (typing, clicking, filtering). That's why the
expensive steps (loading data, loading the model) are cached: they run once
and are reused across re-runs. User choices that must survive re-runs
(the query, the selected movie) live in st.session_state.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

from data_prep import load_data
from metrics import METRIC_LABELS, SPLIT_OPTIONS, filter_consumption, monthly_series, summarize
from search import load_embeddings, load_model, search

st.set_page_config(page_title="Movie Intelligence Explorer", page_icon="🎬", layout="wide")

EXAMPLE_QUERIES = [
    "dark psychological thrillers about obsession",
    "family movies about overcoming loss",
    "animated adventures",
    "movies about artificial intelligence",
]

PLATFORM_TYPE_LABELS = {
    "svod": "Subscription (SVOD)",
    "avod": "Free with ads (AVOD)",
}


# ---------- Cached loading (runs once, reused on every re-run) ----------

@st.cache_data
def get_data():
    movies, availability, consumption, snapshot = load_data()
    embeddings = load_embeddings(movies)
    return movies, availability, consumption, snapshot, embeddings


@st.cache_resource
def get_model():
    # cache_resource: one shared model object, not copied on each re-run.
    return load_model()


# ---------- Small helpers ----------

def show(value, fallback="Not available"):
    """Return a display string, with a fallback for missing values."""
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
        return fallback
    return str(value)


def fmt_number(value, suffix=""):
    """1234.0 -> '1,234' (plus optional suffix). Missing values -> 'N/A'."""
    if pd.isna(value):
        return "N/A"
    return f"{value:,.0f}{suffix}"


def poster_url(url, width=300):
    """Ask the image server for a smaller version of the poster (faster loading).

    IMDb/Amazon image URLs accept a size hint like '._V1_SX300_.'.
    If the URL doesn't follow that pattern, it is returned unchanged.
    """
    if not isinstance(url, str) or not url:
        return None
    return url.replace("._V1_.", f"._V1_SX{width}_.")


def select_movie(title_id):
    st.session_state.selected_id = title_id


def use_example(query):
    st.session_state.query = query
    st.session_state.selected_id = None


# ---------- Load everything ----------

with st.spinner("Loading movies and search model..."):
    movies, availability, consumption, snapshot, embeddings = get_data()
    model = get_model()

st.session_state.setdefault("query", "")
st.session_state.setdefault("selected_id", None)

all_genres = sorted({g for genres in movies["genre_list"] for g in genres})
consumption_countries = sorted(consumption["country"].unique())
consumption_platforms = sorted(consumption["platform"].unique())
snapshot_label = pd.to_datetime(snapshot).strftime("%B %Y")


# ---------- Sidebar: filters ----------

with st.sidebar:
    st.header("Filters")
    selected_genres = st.multiselect(
        "Genres",
        all_genres,
        help="Shows movies that have at least one of the selected genres.",
    )
    top_k = st.slider("Number of results", min_value=5, max_value=30, value=10, step=5)
    st.caption(
        f"{len(movies):,} movies. Streaming availability as of {snapshot_label}."
    )


# ---------- Search box ----------

st.title("🎬 Movie Intelligence Explorer")
st.text_input(
    "Describe the kind of movie you're looking for",
    key="query",
    placeholder="e.g. a lonely hitman losing his nerve",
    help="Search works by meaning, not exact words. English works best.",
)

st.caption("Try an example:")
example_cols = st.columns(len(EXAMPLE_QUERIES))
for col, example in zip(example_cols, EXAMPLE_QUERIES):
    col.button(example, on_click=use_example, args=(example,), width="stretch")

query = st.session_state.query.strip()

if not query:
    st.info("Type a description above or pick an example to start exploring.")
    st.stop()  # nothing else to render until there is a query

results = search(query, model, embeddings, movies, top_k=top_k, genres=selected_genres)

if results.empty:
    st.warning(
        "No movies match these filters. Try removing a genre or rephrasing your search."
    )
    st.stop()

# If the selected movie is not among the current results, select the best match.
if st.session_state.selected_id not in set(results["title_id"]):
    st.session_state.selected_id = results.loc[0, "title_id"]


# ---------- Layout: results (left) and details (right) ----------

results_col, detail_col = st.columns([2, 3], gap="large")

with results_col:
    st.subheader(f"Results for “{query}”")
    for _, movie in results.iterrows():
        is_selected = movie["title_id"] == st.session_state.selected_id
        with st.container(border=True):
            img_col, text_col = st.columns([1, 3])
            url = poster_url(movie["image_url"], width=150)
            if url:
                img_col.image(url, width="stretch")
            text_col.markdown(f"**{movie['original_title']}** ({movie['year']})")
            text_col.caption(show(movie["genres"], "Genres not available"))
            text_col.caption(
                f"⭐ {show(movie['rating_value'])} / 10 ({fmt_number(movie['rating_vote_count'])} votes)"
            )
            text_col.caption(f"Similarity: {movie['score']:.2f}")
            text_col.button(
                "Viewing" if is_selected else "View details",
                key=f"view_{movie['title_id']}",
                on_click=select_movie,
                args=(movie["title_id"],),
                disabled=is_selected,
            )

with detail_col:
    movie = movies.loc[movies["title_id"] == st.session_state.selected_id].iloc[0]

    # --- Movie information ---
    st.header(f"{movie['original_title']} ({movie['year']})")
    poster_col, info_col = st.columns([1, 2])
    url = poster_url(movie["image_url"], width=400)
    if url:
        poster_col.image(url, width="stretch")

    with info_col:
        m1, m2, m3 = st.columns(3)
        m1.metric("Rating", f"{show(movie['rating_value'])} / 10")
        m2.metric("Votes", fmt_number(movie["rating_vote_count"]))
        m3.metric("Runtime", fmt_number(movie["runtime_minutes"], " min"))
        st.markdown(f"**Genres:** {show(movie['genres'])}")
        st.markdown(f"**Director:** {show(movie['directors'])}")
        st.markdown(f"**Cast:** {show(movie['principal_cast'])}")
        if isinstance(movie["imdb_url"], str):
            st.markdown(f"[View on IMDb]({movie['imdb_url']})")

    st.markdown("**Plot**")
    st.write(show(movie["plot_summary"], "No plot summary available."))

    # --- Streaming availability (dataset B, joined only with movies) ---
    st.subheader(f"Where to stream ({snapshot_label})")
    movie_avail = availability[availability["title_id"] == movie["title_id"]]

    if movie_avail.empty:
        st.info("This movie is not available on any tracked streaming platform.")
    else:
        n_countries = movie_avail["country"].nunique()
        n_platforms = movie_avail["platform"].nunique()
        st.caption(
            f"Available in {n_countries} {'country' if n_countries == 1 else 'countries'} "
            f"on {n_platforms} {'platform' if n_platforms == 1 else 'platforms'}."
        )
        table = (
            movie_avail.assign(
                platform_type=movie_avail["platform_type"].map(PLATFORM_TYPE_LABELS)
                .fillna(movie_avail["platform_type"]),
                original_flag=movie_avail["original_flag"].replace("Not An Original", "—"),
            )
            .sort_values(["country", "platform"])
            [["country", "platform", "platform_type", "original_flag"]]
            .rename(columns={
                "country": "Country",
                "platform": "Platform",
                "platform_type": "Type",
                "original_flag": "Original / exclusive",
            })
        )
        st.dataframe(table, hide_index=True, width="stretch")

    # --- Consumption performance (dataset C, joined only with movies) ---
    st.subheader("Consumption performance")
    movie_cons = filter_consumption(consumption, movie["title_id"])

    if movie_cons.empty:
        st.info("No consumption data recorded for this movie.")
    else:
        f1, f2 = st.columns(2)
        countries = f1.multiselect(
            "Countries", consumption_countries, key="cons_countries", placeholder="All countries"
        )
        platforms = f2.multiselect(
            "Platforms", consumption_platforms, key="cons_platforms", placeholder="All platforms"
        )
        o1, o2 = st.columns(2)
        metric = o1.radio(
            "Metric", list(METRIC_LABELS), format_func=METRIC_LABELS.get,
            horizontal=True, key="cons_metric",
        )
        split_label = o2.radio("Show", list(SPLIT_OPTIONS), horizontal=True, key="cons_split")
        split = SPLIT_OPTIONS[split_label]

        filtered = filter_consumption(consumption, movie["title_id"], countries, platforms)

        if filtered.empty:
            st.warning("No consumption data for this combination. Try other countries or platforms.")
        else:
            stats = summarize(filtered)
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total streams", fmt_number(stats["total_streams"]))
            k2.metric("Total minutes", fmt_number(stats["total_minutes"]))
            k3.metric(
                "Minutes per stream",
                fmt_number(stats["minutes_per_stream"]) if stats["minutes_per_stream"] is not None else "N/A",
                help="Total minutes ÷ total streams.",
            )
            k4.metric(
                "Peak month",
                stats["peak_month"].strftime("%b %Y") if stats["peak_month"] is not None else "N/A",
                help="Month with the most streams.",
            )

            series = monthly_series(filtered, metric, split)
            fig = px.line(
                series,
                x="month",
                y=metric,
                color=split,
                markers=True,
                labels={"month": "Month", metric: METRIC_LABELS[metric],
                        "country": "Country", "platform": "Platform"},
            )
            fig.update_layout(
                height=360,
                margin=dict(l=0, r=0, t=10, b=0),
                hovermode="x unified",
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig, width="stretch")

        st.caption(
            "Gaps in the line are months with no records (not shown as zero). "
            f"Consumption data covers {len(consumption_countries)} countries "
            f"({', '.join(consumption_countries)}) and {len(consumption_platforms)} platforms, "
            "while availability covers more countries."
        )