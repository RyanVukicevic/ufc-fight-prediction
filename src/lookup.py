import difflib

import pandas as pd


def normalize_name(name: str) -> str:
    return " ".join(str(name).lower().replace("-", " ").split())


def find_fighter_in_fighters(name: str, fighters: pd.DataFrame, n_matches: int = 10) -> pd.DataFrame:
    """Find exact and close fighter-name matches in the fighters table."""
    query = normalize_name(name)
    out = fighters.copy()
    out["_lookup_name"] = out["fighter"].map(normalize_name)

    exact = out[out["_lookup_name"] == query].copy()
    if not exact.empty:
        exact["match_type"] = "exact"
        exact["similarity"] = 1.0
        return exact.drop(columns=["_lookup_name"])

    contains = out[out["_lookup_name"].str.contains(query, regex=False, na=False)].copy()
    contains["match_type"] = "contains"
    contains["similarity"] = 1.0

    choices = out["_lookup_name"].tolist()
    close_names = difflib.get_close_matches(query, choices, n=n_matches, cutoff=0.70)
    close = out[out["_lookup_name"].isin(close_names)].copy()
    close["match_type"] = "close"
    close["similarity"] = close["_lookup_name"].map(
        lambda x: difflib.SequenceMatcher(None, query, x).ratio()
    )

    matches = pd.concat([contains, close], ignore_index=True)
    matches = matches.drop_duplicates(subset=["fighter_url"])
    matches = matches.sort_values(["similarity", "fighter"], ascending=[False, True])

    return matches.drop(columns=["_lookup_name"]).head(n_matches)


def find_fighter_in_fighter_stats(name: str, fighter_stats: pd.DataFrame, n_matches: int = 10) -> pd.DataFrame:
    """Find fighter snapshots by name in fighter_stats."""
    query = normalize_name(name)
    out = fighter_stats.copy()
    out["_lookup_name"] = out["fighter"].map(normalize_name)

    matches = out[out["_lookup_name"].str.contains(query, regex=False, na=False)].copy()
    if matches.empty:
        choices = out["_lookup_name"].drop_duplicates().tolist()
        close_names = difflib.get_close_matches(query, choices, n=n_matches, cutoff=0.85)
        matches = out[out["_lookup_name"].isin(close_names)].copy()

    return matches.drop(columns=["_lookup_name"]).sort_values(["fighter", "date", "bout_url"]).tail(n_matches)


def find_fighter_in_full_fight_stats(name: str, full_fight_stats: pd.DataFrame, n_matches: int = 10) -> pd.DataFrame:
    """Find fight-level rows where the fighter appears on either side."""
    query = normalize_name(name)
    out = full_fight_stats.copy()
    a_name = out["fighter_a"].map(normalize_name)
    b_name = out["fighter_b"].map(normalize_name)

    matches = out[
        a_name.str.contains(query, regex=False, na=False)
        | b_name.str.contains(query, regex=False, na=False)
    ].copy()

    return matches.sort_values(["date", "bout_url"]).tail(n_matches)
