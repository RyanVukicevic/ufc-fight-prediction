import pandas as pd
import numpy as np
from src.raw_data import load_raw_data
from src.elo import add_elo_features


# FUNCTION CALLS for load_cleaned_data (in order), which is the "interface"
# and is the only one that i need to call
#
#     load_cleaned_data
#         make_fights
#             extract_weightclass
#             clean_method
#         make_fighters
#         apply_name_aliases
#         add_fighter_urls_simple
#         make_fights_long
#         build_fighter_fight_stats
#             add_fight_indicators
#             compute_entering_cumsums
#             add_entering_rates_and_layoff
#         add_static_attrs_and_age_at_fight
#         build_model_fights
#             impute_height_and_reach_by_weightclass
#         add_elo_features
#         make_symmetric_delta_dataset


def extract_weightclass(text):
    """helper for make_fights(), cleans the weightclasses of fights["weightclass"] so that only those in
    valid_weightclasses list are present in the column"""

    valid_weightclasses = [
        "Flyweight",
        "Bantamweight",
        "Featherweight",
        "Lightweight",
        "Welterweight",
        "Middleweight",
        "Light Heavyweight",
        "Heavyweight",
    ]

    for wc in valid_weightclasses:
        if wc in text:
            return wc
    return None


def clean_method(method):
    """helper for make_fights(), cleans the methods of victory of fights["method"] so that only those in
    valid_methods list are present in the column"""

    valid_methods = ["KO/TKO", "Submission", "Decision", "DQ"]

    if pd.isna(method):
        return None

    m = method.lower()

    if "ko" in m or "tko" in method or "doctor" in m:
        return "KO/TKO"

    if "submission" in m:
        return "Submission"

    if "decision" in m:
        return "Decision"

    if "dq" in m:
        return "DQ"

    return None


def make_fights(raw) -> pd.DataFrame:
    """three-way merge from event_details, fight_details, and results datframes
    this is to obtain a record of all fights, with relevant info
    output: fights dataframe, a fully cleaned record"""

    event_details = raw["event_details"].copy()
    fight_details = raw["fight_details"].copy()
    results = raw["results"].copy()

    event_details.rename(columns={"url": "event_url"}, inplace=True)
    fight_details.rename(columns={"url": "fight_url"}, inplace=True)
    results.rename(columns={"url": "fight_url"}, inplace=True)

    intermediate = pd.merge(event_details, fight_details, how="inner", on="event")
    fights = pd.merge(intermediate, results, on="fight_url", how="inner")

    fights.rename(columns={"event_y": "event", "bout_y": "bout"}, inplace=True)
    fights.drop(columns=["event_x", "bout_x"], inplace=True)

    fights["date"] = pd.to_datetime(fights["date"])

    acceptable_time_formats = ["3 Rnd (5-5-5)", "5 Rnd (5-5-5-5-5)"]
    fights = fights[fights["time format"].isin(acceptable_time_formats)].copy()

    time_format_map = {
        "3 Rnd (5-5-5)": 3,
        "5 Rnd (5-5-5-5-5)": 5,
    }
    fights["scheduled_rounds"] = fights["time format"].map(time_format_map)

    fights = fights[~fights["weightclass"].str.contains("Women", na=False)].copy()

    fights["weightclass_clean"] = fights["weightclass"].apply(extract_weightclass)
    fights = fights[fights["weightclass_clean"].notna()].copy()

    fights = fights[fights["outcome"] != "D/D"].copy()

    fights["victory_method"] = fights["method"].apply(clean_method)
    fights = fights[fights["victory_method"].notna()].copy()

    fights[["fighter_a", "fighter_b"]] = fights["bout"].str.split(" vs. ", expand=True)

    fights["winner"] = np.where(
        fights["outcome"] == "W/L",
        fights["fighter_a"],
        fights["fighter_b"],
    )

    fights["loser"] = np.where(
        fights["outcome"] == "W/L",
        fights["fighter_b"],
        fights["fighter_a"],
    )

    methods = ["Submission", "KO/TKO"]
    fights["finish"] = fights["victory_method"].isin(methods)

    fights["time_in_round_seconds"] = (
        fights["time"].str.split(":").str[0].astype(int) * 60
        + fights["time"].str.split(":").str[1].astype(int)
    )

    fights["time_elapsed_seconds"] = (
        (fights["round"] - 1) * 300 + fights["time_in_round_seconds"]
    )

    fights["max_fight_seconds"] = fights["scheduled_rounds"] * 300
    fights["pct_of_fight_completed"] = (
        fights["time_elapsed_seconds"] / fights["max_fight_seconds"]
    ).round(2)

    fights["ppv"] = ~fights["event"].str.lower().str.contains("fight night")

    drop_cols = ["time", "time format", "weightclass", "details", "referee", "outcome"]
    rename_cols = {
        "victory_method": "method",
        "weightclass_clean": "weightclass",
        "fight_url": "bout_url",
    }

    fights.drop(columns=drop_cols, inplace=True)
    fights.rename(columns=rename_cols, inplace=True)

    fights = fights[[
        "bout",
        "event",
        "date",
        "location",
        "fighter_a",
        "fighter_b",
        "winner",
        "loser",
        "weightclass",
        "finish",
        "method",
        "round",
        "ppv",
        "scheduled_rounds",
        "time_elapsed_seconds",
        "max_fight_seconds",
        "pct_of_fight_completed",
        "bout_url",
        "event_url",
    ]]

    return fights



def make_fighters(raw):
    """merge and clean dataframes fighters, tott for a complete fighters dataframe with cleaned stats
    output: fighters dataframe"""

    fighters = raw["fighters"].copy()
    tott = raw["tott"].copy()

    fighters["fighter"] = (
        raw["fighters"]["first"].str.strip() + " " + raw["fighters"]["last"].str.strip()
    )

    fighters = tott.merge(fighters, on="url")
    fighters.replace("--", np.nan, inplace=True)

    fighters["dob"] = pd.to_datetime(fighters["dob"])

    fighters[["feet", "inch"]] = fighters["height"].str.extract(r"(\d+)'\s*(\d+)?")
    fighters["feet"] = pd.to_numeric(fighters["feet"], errors="coerce")
    fighters["inch"] = pd.to_numeric(fighters["inch"], errors="coerce").fillna(0)
    fighters["height"] = fighters["feet"] * 12 + fighters["inch"]

    fighters["reach"] = pd.to_numeric(
        fighters["reach"].str.replace('"', "", regex=False),
        errors="coerce",
    )

    drop_cols = ["first", "last", "nickname", "fighter_y", "feet", "inch"]
    rename_cols = {"fighter_x": "fighter", "url": "fighter_url"}

    fighters.drop(columns=drop_cols, inplace=True)
    fighters.rename(columns=rename_cols, inplace=True)

    fighters = fighters[["fighter", "weight", "height", "reach", "stance", "dob", "fighter_url"]]

    return fighters



def apply_name_aliases(fights: pd.DataFrame) -> pd.DataFrame:
    """helper that fixes the rare broken names where the name in fights[bout] is not in fighters
    alias holds the problematic names as keys, values are the fixed version so the mapping is 1-to-1
    called before add_fighter_urls_simple()"""
    fights = fights.copy()

    alias = {
        "Waldo Cortes Acosta": "Waldo Cortes-Acosta",
        "Zach Reese": "Zachary Reese",
        "Shem Rock": "Shaqueme Rock",
        "Michael Aswell Jr.": "Michael Aswell",
        "Rafael Cerquiera": "Rafael Cerqueira",
    }

    fights["fighter_a"] = fights["fighter_a"].replace(alias)
    fights["fighter_b"] = fights["fighter_b"].replace(alias)
    fights["bout"] = fights["fighter_a"] + " vs. " + fights["fighter_b"]
    fights["winner"] = fights["winner"].replace(alias)
    fights["loser"] = fights["loser"].replace(alias)

    return fights



def add_fighter_urls_simple(fights: pd.DataFrame, fighters: pd.DataFrame) -> pd.DataFrame:
    fights_out = fights.copy()

    name_to_url = fighters.set_index("fighter")["fighter_url"].to_dict()

    fights_out["fighter_a_url"] = fights_out["fighter_a"].map(name_to_url)
    fights_out["fighter_b_url"] = fights_out["fighter_b"].map(name_to_url)

    return fights_out



def make_fights_long(fights: pd.DataFrame) -> pd.DataFrame:
    fights = fights.copy()

    base_cols = [
        "bout_url",
        "date",
        "weightclass",
        "scheduled_rounds",
        "finish",
        "method",
        "time_elapsed_seconds",
        "fighter_a",
        "fighter_b",
        "winner",
        "ppv",
        "fighter_a_url",
        "fighter_b_url",
    ]

    a_side = fights[base_cols].copy()
    a_side["fighter_name"] = a_side["fighter_a"]
    a_side["opponent_name"] = a_side["fighter_b"]
    a_side["fighter_url"] = a_side["fighter_a_url"]
    a_side["opponent_url"] = a_side["fighter_b_url"]
    a_side["result"] = (a_side["winner"] == a_side["fighter_a"]).astype(int)
    a_side["corner"] = "A"

    b_side = fights[base_cols].copy()
    b_side["fighter_name"] = b_side["fighter_b"]
    b_side["opponent_name"] = b_side["fighter_a"]
    b_side["fighter_url"] = b_side["fighter_b_url"]
    b_side["opponent_url"] = b_side["fighter_a_url"]
    b_side["result"] = (b_side["winner"] == b_side["fighter_b"]).astype(int)
    b_side["corner"] = "B"

    fights_long = pd.concat([a_side, b_side], ignore_index=True)

    fights_long = fights_long.drop(
        columns=["fighter_a", "fighter_b", "winner", "fighter_a_url", "fighter_b_url"]
    )

    fights_long = fights_long.sort_values(["fighter_url", "date"]).reset_index(drop=True)

    fights_long = fights_long[[
        "bout_url",
        "date",
        "fighter_name",
        "opponent_name",
        "fighter_url",
        "opponent_url",
        "result",
        "weightclass",
        "scheduled_rounds",
        "finish",
        "method",
        "ppv",
        "time_elapsed_seconds",
    ]]

    return fights_long



def add_fight_indicators(fights_long: pd.DataFrame) -> pd.DataFrame:
    df = fights_long.copy()

    df["is_win"] = df["result"].astype(int)
    df["is_loss"] = (1 - df["is_win"]).astype(int)

    df["is_finish"] = df["finish"].astype(int)
    df["is_decision"] = (df["method"] == "Decision").astype(int)

    df["is_ko"] = (df["method"] == "KO/TKO").astype(int)
    df["is_sub"] = (df["method"] == "Submission").astype(int)
    df["is_dq"] = (df["method"] == "DQ").astype(int)

    df["finish_win"] = (df["is_win"] & df["is_finish"]).astype(int)
    df["finish_loss"] = (df["is_loss"] & df["is_finish"]).astype(int)

    df["ko_win"] = (df["is_win"] & df["is_ko"]).astype(int)
    df["ko_loss"] = (df["is_loss"] & df["is_ko"]).astype(int)

    df["sub_win"] = (df["is_win"] & df["is_sub"]).astype(int)
    df["sub_loss"] = (df["is_loss"] & df["is_sub"]).astype(int)

    df["is_5_round"] = (df["scheduled_rounds"] == 5).astype(int)

    return df



def compute_entering_cumsums(
    fights_long_with_indicators: pd.DataFrame,
    group_key: str = "fighter_url",
    date_col: str = "date",
) -> pd.DataFrame:
    df = fights_long_with_indicators.copy()

    df = df.sort_values([group_key, date_col]).reset_index(drop=True)
    g = df.groupby(group_key, sort=False)

    df["fights_so_far"] = g.cumcount() + 1
    df["wins_so_far"] = g["is_win"].cumsum()
    df["losses_so_far"] = g["is_loss"].cumsum()

    df["finish_wins_so_far"] = g["finish_win"].cumsum()
    df["finish_losses_so_far"] = g["finish_loss"].cumsum()

    df["ko_wins_so_far"] = g["ko_win"].cumsum()
    df["ko_losses_so_far"] = g["ko_loss"].cumsum()

    df["sub_wins_so_far"] = g["sub_win"].cumsum()
    df["sub_losses_so_far"] = g["sub_loss"].cumsum()

    df["five_round_fights_so_far"] = g["is_5_round"].cumsum()

    df["avg_fight_time_so_far"] = (
        g["time_elapsed_seconds"].expanding().mean().reset_index(level=0, drop=True)
    )

    to_shift = [
        "fights_so_far",
        "wins_so_far",
        "losses_so_far",
        "finish_wins_so_far",
        "finish_losses_so_far",
        "ko_wins_so_far",
        "ko_losses_so_far",
        "sub_wins_so_far",
        "sub_losses_so_far",
        "five_round_fights_so_far",
        "avg_fight_time_so_far",
    ]

    for c in to_shift:
        df[c.replace("_so_far", "_entering")] = g[c].shift(1)

    count_cols = [c for c in df.columns if c.endswith("_entering") and c != "avg_fight_time_entering"]
    df[count_cols] = df[count_cols].fillna(0)

    overall_mean_time = df["time_elapsed_seconds"].mean()
    df["avg_fight_time_entering"] = df["avg_fight_time_entering"].fillna(overall_mean_time)

    return df



def add_entering_rates_and_layoff(
    df: pd.DataFrame,
    group_key: str = "fighter_url",
    date_col: str = "date",
) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby(group_key, sort=False)

    out["days_since_last_fight"] = g[date_col].diff().dt.days
    out["days_since_last_fight"] = out["days_since_last_fight"].fillna(-1)

    def safe_div(num, den):
        den = den.replace(0, np.nan)
        return (num / den).fillna(0)

    fights_e = out["fights_entering"]

    out["win_rate_entering"] = safe_div(out["wins_entering"], fights_e)
    out["finish_win_rate_entering"] = safe_div(out["finish_wins_entering"], fights_e)
    out["ko_win_rate_entering"] = safe_div(out["ko_wins_entering"], fights_e)
    out["sub_win_rate_entering"] = safe_div(out["sub_wins_entering"], fights_e)

    out["finish_loss_rate_entering"] = safe_div(out["finish_losses_entering"], fights_e)
    out["ko_loss_rate_entering"] = safe_div(out["ko_losses_entering"], fights_e)
    out["sub_loss_rate_entering"] = safe_div(out["sub_losses_entering"], fights_e)

    out["five_round_rate_entering"] = safe_div(out["five_round_fights_entering"], fights_e)

    return out



def build_fighter_fight_stats(fights_long: pd.DataFrame) -> pd.DataFrame:
    df = add_fight_indicators(fights_long)
    df = compute_entering_cumsums(df)
    df = add_entering_rates_and_layoff(df)

    keep_cols = [
        "bout_url",
        "date",
        "fighter_name",
        "fighter_url",
        "opponent_name",
        "opponent_url",
        "result",
        "weightclass",
        "scheduled_rounds",
        "finish",
        "method",
        "time_elapsed_seconds",
        "days_since_last_fight",
        "fights_entering",
        "wins_entering",
        "losses_entering",
        "finish_wins_entering",
        "finish_losses_entering",
        "ko_wins_entering",
        "ko_losses_entering",
        "sub_wins_entering",
        "sub_losses_entering",
        "five_round_fights_entering",
        "avg_fight_time_entering",
        "win_rate_entering",
        "finish_win_rate_entering",
        "ko_win_rate_entering",
        "sub_win_rate_entering",
        "five_round_rate_entering",
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols].copy()



def add_static_attrs_and_age_at_fight(
    fighter_fight_stats: pd.DataFrame,
    fighters: pd.DataFrame,
    age_impute: int = 30,
) -> pd.DataFrame:
    out = fighter_fight_stats.copy()

    static = fighters[["fighter_url", "dob", "height", "reach"]].copy()
    static["dob"] = pd.to_datetime(static["dob"], errors="coerce")

    out = out.merge(static, on="fighter_url", how="left")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    dob_month = out["dob"].dt.month
    dob_day = out["dob"].dt.day
    fight_month = out["date"].dt.month
    fight_day = out["date"].dt.day

    before_bday = (fight_month < dob_month) | ((fight_month == dob_month) & (fight_day < dob_day))

    out["age_at_fight"] = out["date"].dt.year - out["dob"].dt.year - before_bday.astype("Int64")

    out["age_missing"] = out["age_at_fight"].isna().astype(int)
    out["age_at_fight"] = out["age_at_fight"].fillna(age_impute).astype(int)

    out = out.drop(columns=["dob"])

    return out



def impute_height_and_reach_by_weightclass(model_fights: pd.DataFrame) -> pd.DataFrame:
    """helper for build_model_fights()"""

    df = model_fights.copy()

    height_long = pd.concat([
        df[["weightclass", "a_height"]].rename(columns={"a_height": "height"}),
        df[["weightclass", "b_height"]].rename(columns={"b_height": "height"}),
    ])

    median_height = height_long.groupby("weightclass")["height"].median()

    df["a_height"] = df["a_height"].fillna(df["weightclass"].map(median_height))
    df["b_height"] = df["b_height"].fillna(df["weightclass"].map(median_height))

    reach_long = pd.concat([
        df[["weightclass", "a_reach"]].rename(columns={"a_reach": "reach"}),
        df[["weightclass", "b_reach"]].rename(columns={"b_reach": "reach"}),
    ])

    median_reach = reach_long.groupby("weightclass")["reach"].median()

    df["a_reach"] = df["a_reach"].fillna(df["weightclass"].map(median_reach))
    df["b_reach"] = df["b_reach"].fillna(df["weightclass"].map(median_reach))

    return df



def build_model_fights(fights: pd.DataFrame, fighter_fight_stats: pd.DataFrame) -> pd.DataFrame:
    fights_df = fights.copy()
    stats = fighter_fight_stats.copy()

    for c in ["bout_url", "fighter_a_url", "fighter_b_url"]:
        if c in fights_df.columns and fights_df[c].dtype == "object":
            fights_df[c] = fights_df[c].str.strip()

    for c in ["bout_url", "fighter_url"]:
        if c in stats.columns and stats[c].dtype == "object":
            stats[c] = stats[c].str.strip()

    y_map = stats[["bout_url", "fighter_url", "result"]].rename(
        columns={"fighter_url": "fighter_a_url", "result": "y"}
    )
    fights_df = fights_df.merge(
        y_map,
        on=["bout_url", "fighter_a_url"],
        how="left",
    )

    id_cols = {
        "bout_url",
        "fighter_url",
        "fighter_name",
        "opponent_url",
        "opponent_name",
        "date",
        "result",
        "y",
        "weightclass",
        "scheduled_rounds",
        "finish",
        "method",
        "time_elapsed_seconds",
    }
    stat_cols = [c for c in stats.columns if c not in id_cols]
    stat_cols.remove("age_missing")

    a_stats = stats[["bout_url", "fighter_url"] + stat_cols].copy()
    a_stats = a_stats.rename(columns={c: f"a_{c}" for c in stat_cols})

    fights_df = fights_df.merge(
        a_stats,
        left_on=["bout_url", "fighter_a_url"],
        right_on=["bout_url", "fighter_url"],
        how="left",
    ).drop(columns=["fighter_url"])

    b_stats = stats[["bout_url", "fighter_url"] + stat_cols].copy()
    b_stats = b_stats.rename(columns={c: f"b_{c}" for c in stat_cols})

    fights_df = fights_df.merge(
        b_stats,
        left_on=["bout_url", "fighter_b_url"],
        right_on=["bout_url", "fighter_url"],
        how="left",
    ).drop(columns=["fighter_url"])

    fights_df = impute_height_and_reach_by_weightclass(fights_df)

    return fights_df



def make_symmetric_delta_dataset(full_fight_stats: pd.DataFrame) -> pd.DataFrame:
    df1 = full_fight_stats.copy()

    a_cols = [c for c in df1.columns if c.startswith("a_")]
    b_cols = [c for c in df1.columns if c.startswith("b_")]

    a_bases = {c[2:] for c in a_cols}
    b_bases = {c[2:] for c in b_cols}
    shared_bases = sorted(a_bases.intersection(b_bases))

    delta_cols = []
    for base in shared_bases:
        a = f"a_{base}"
        b = f"b_{base}"
        if pd.api.types.is_numeric_dtype(df1[a]) and pd.api.types.is_numeric_dtype(df1[b]):
            d = f"delta_{base}"
            df1[d] = df1[a] - df1[b]
            delta_cols.append(d)

    df2 = df1.copy()

    for left, right in [
        ("fighter_a", "fighter_b"),
        ("fighter_a_url", "fighter_b_url"),
        ("winner", "loser"),
    ]:
        if left in df2.columns and right in df2.columns:
            df2[left], df2[right] = df1[right].values, df1[left].values

    for base in shared_bases:
        a = f"a_{base}"
        b = f"b_{base}"
        df2[a], df2[b] = df1[b].values, df1[a].values

    if delta_cols:
        df2[delta_cols] = -df2[delta_cols]

    if "y" in df2.columns:
        df2["y"] = 1 - df2["y"]

    return pd.concat([df1, df2], ignore_index=True)



def make_x_and_y(full_sym: pd.DataFrame):
    df = full_sym.copy()

    df = df.sort_values(["date", "bout_url"], kind="mergesort").reset_index(drop=True)

    df_sorted = df.sort_values(["bout_url", "date"], kind="mergesort").reset_index(drop=True)
    assert (df_sorted["bout_url"].shift(1) == df_sorted["bout_url"]).iloc[1::2].all()

    df = df_sorted.copy()

    delta_cols = [c for c in df.columns if c.startswith("delta_")]
    base_cols = [c for c in ["date", "ppv", "scheduled_rounds", "weightclass"] if c in df.columns]

    X = df[base_cols + delta_cols].copy()
    y = df["y"].astype(int)

    if "weightclass" in X.columns:
        X = pd.get_dummies(X, columns=["weightclass"], drop_first=True)

    return X, y



def train_test_split_xy(x, y, train_size=0.8):
    y = pd.Series(y).reset_index(drop=True)
    x = x.reset_index(drop=True)

    n = len(x)
    assert n == len(y), "X and y must have same length"
    assert n % 2 == 0, "need even number of rows (2 per fight)"

    n_fights = n // 2
    cut = int(n_fights * train_size)
    split_row = 2 * cut

    x_train = x.iloc[:split_row].reset_index(drop=True)
    y_train = y.iloc[:split_row].reset_index(drop=True)

    x_test = x.iloc[split_row:].reset_index(drop=True)
    y_test = y.iloc[split_row:].reset_index(drop=True)

    return x_train, y_train, x_test, y_test



def load_cleaned_data(raw):
    """calls many helper functions to clean data into multiple useful dataframes for analysis, bookkeeping,
    and training
    output:
    fighters df for tracking all fighters
    full_fight_stats for all fights with snapshots of both fighters with a_ and b_ stats
    full_sym, each row being doubled and looked at from either fighter pov, accompanied with delta_ stats too
    """

    fights = make_fights(raw)
    fighters = make_fighters(raw)

    fights_with_urls = apply_name_aliases(fights)
    fights_with_urls = add_fighter_urls_simple(fights_with_urls, fighters)
    fights = fights_with_urls.copy()

    fights_long = make_fights_long(fights)

    fighter_fight_stats = build_fighter_fight_stats(fights_long)
    fighter_fight_stats = add_static_attrs_and_age_at_fight(fighter_fight_stats, fighters)

    full_fight_stats = build_model_fights(fights=fights, fighter_fight_stats=fighter_fight_stats)

    full_fight_stats = add_elo_features(
        full_fight_stats,
        fighter_a_col="fighter_a_url",
        fighter_b_col="fighter_b_url",
        y_col="y",
        date_col="date",
        method_col="method",
        round_col="round",
        scheduled_rounds_col="scheduled_rounds",
    )

    full_sym = make_symmetric_delta_dataset(full_fight_stats)

    cleaned = {
        "fighters": fighters,
        "full_fight_stats": full_fight_stats,
        "full_sym": full_sym,
    }

    return cleaned
