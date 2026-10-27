"""
Loading, cleaning and validation of the three datasets.

Resulting tables (all keyed by title_id):
- movies:       1 row per movie                              (dataset A)
- availability: 1 row per movie × country × platform         (dataset B)
- consumption:  1 row per movie × month × country × platform (dataset C)

"""
from pathlib import Path

import pandas as pd

# data/ folder next to this file, regardless of where it is run from.
DATA_DIR = Path(__file__).parent / "data"


def load_data():
    """Return (movies, availability, consumption, availability_snapshot)."""
    # --- A: movies ---
    movies = pd.read_csv(DATA_DIR / "dataset_A.csv")
    movies.columns = movies.columns.str.lower()

    # --- B: availability ---
    availability = pd.read_csv(DATA_DIR / "dataset_B.csv")
    availability.columns = availability.columns.str.lower()
    availability = availability.rename(columns={"imdb_id": "title_id"}, errors="raise")

    # Availability snapshot date: must be unique.
    assert availability["month_year"].nunique() == 1, "availability: more than one snapshot date"
    availability_snapshot = availability["month_year"].iloc[0]
    availability = availability.drop(columns=["ampere_title", "month_year"])

    # --- C: consumption ---
    consumption = pd.read_csv(DATA_DIR / "dataset_C.csv")
    consumption.columns = consumption.columns.str.lower()
    consumption = consumption.rename(columns={"imdb_id": "title_id"}, errors="raise")
    consumption["month"] = pd.to_datetime(consumption["month"])

    validate(movies, availability, consumption)
    return movies, availability, consumption, availability_snapshot


def validate(movies, availability, consumption):
    """Check grains, referential integrity and valid values."""
    # Grain 
    assert not movies.duplicated(subset=["title_id"]).any(), "movies: duplicated title_id"
    assert not availability.duplicated(
        subset=["title_id", "country", "platform"]
    ).any(), "availability: duplicated grain"
    assert not consumption.duplicated(
        subset=["title_id", "month", "country", "platform"]
    ).any(), "consumption: duplicated grain"

    # platform_type depends only on platform, so it is not part of the key
    assert (
        availability.groupby("platform")["platform_type"].nunique() <= 1
    ).all(), "availability: a platform has more than one type"

    # Referential integrity 
    assert availability["title_id"].isin(movies["title_id"]).all(), \
        "availability: movies not found in movies table"
    assert consumption["title_id"].isin(movies["title_id"]).all(), \
        "consumption: movies not found in movies table"

    # Valid values 
    assert (consumption["streams"] >= 0).all(), "consumption: negative streams"
    assert (consumption["total_minutes"] >= 0).all(), "consumption: negative total_minutes"


if __name__ == "__main__":
    movies, availability, consumption, snapshot = load_data()
    print("Validations OK ✔")
    print("movies:      ", movies.shape)
    print("availability:", availability.shape)
    print("consumption: ", consumption.shape)
    print("snapshot:    ", snapshot)