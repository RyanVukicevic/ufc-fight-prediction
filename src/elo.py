
import math
from collections import defaultdict, deque
import pandas as pd


def _expected_score(r_a: float, r_b: float, scale: float) -> float:
    # standard elo logistic curve
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / scale))


def _mov_weight(method: str, round_num: float) -> float:
    # simple, monotonic "how convincing was the win" multiplier
    # keep it coarse on purpose (robust + easy to tune later)
    if method is None or (isinstance(method, float) and pd.isna(method)):
        return 1.0

    m = str(method).lower()
    try:
        r = int(round_num)
    except Exception:
        r = None

    # decisions are "lower margin" than finishes
    if "decision" in m:
        # split/majority are slightly weaker signals than unanimous
        if "split" in m or "majority" in m:
            return 1.03
        return 1.10

    # submissions / ko-tko: more weight, earlier rounds more weight
    if "sub" in m:
        base = 1.25
    elif "ko" in m or "tko" in m or "doctor" in m:
        base = 1.30
    else:
        base = 1.0

    if r is None:
        return base

    # decay by round (round 1 biggest)
    # 1 -> 1.00, 2 -> 0.96, 3 -> 0.92, 4 -> 0.88, 5 -> 0.84
    round_factor = max(0.84, 1.00 - 0.04 * (r - 1))
    return base * round_factor


def add_elo_features(
    fights_df: pd.DataFrame,
    *,
    fighter_a_col: str = "fighter_a_url",
    fighter_b_col: str = "fighter_b_url",
    y_col: str = "y",
    date_col: str = "date",
    method_col: str = "method",
    round_col: str = "round",
    scheduled_rounds_col: str = "scheduled_rounds",
    scale: float = 400.0,
    k: float = 32.0,
    base_rating: float = 1500.0,
) -> pd.DataFrame:
    """
    Computes Elo-style ratings *chronologically* and adds pre-fight features:

    - a_elo_pre, b_elo_pre
    - a_elo_change_last_3, b_elo_change_last_3
    - a_elo_fights, b_elo_fights

    No leakage:
      For each fight row, features are recorded BEFORE updating from that fight.

    Assumptions:
      - y is from fighter_a's perspective (1 = fighter_a won, 0 = lost)
      - date is datetime-like (already converted in make_fights)
    """

    df = fights_df.copy()

    # stable sort so same-day ordering is deterministic
    sort_cols = [c for c in [date_col, "bout_url"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    ratings = defaultdict(lambda: base_rating)
    fights_count = defaultdict(int)

    # track last 3 pre-ratings for "form"
    last_pre = defaultdict(lambda: deque(maxlen=3))

    a_pre_list = []
    b_pre_list = []
    a_change3_list = []
    b_change3_list = []
    a_n_list = []
    b_n_list = []

    for i in range(len(df)):
        a_id = df.at[i, fighter_a_col]
        b_id = df.at[i, fighter_b_col]

        # handle missing ids safely
        if pd.isna(a_id) or pd.isna(b_id):
            a_pre = base_rating
            b_pre = base_rating
            a_change3 = 0.0
            b_change3 = 0.0
            a_n = 0
            b_n = 0
            a_pre_list.append(a_pre)
            b_pre_list.append(b_pre)
            a_change3_list.append(a_change3)
            b_change3_list.append(b_change3)
            a_n_list.append(a_n)
            b_n_list.append(b_n)
            continue

        a_id = str(a_id).strip()
        b_id = str(b_id).strip()

        a_pre = float(ratings[a_id])
        b_pre = float(ratings[b_id])

        # form features: delta vs 3 fights ago (0 if <3 fights)
        a_hist = last_pre[a_id]
        b_hist = last_pre[b_id]
        a_change3 = a_pre - (a_hist[0] if len(a_hist) == 3 else a_pre)
        b_change3 = b_pre - (b_hist[0] if len(b_hist) == 3 else b_pre)

        a_n = fights_count[a_id]
        b_n = fights_count[b_id]

        a_pre_list.append(a_pre)
        b_pre_list.append(b_pre)
        a_change3_list.append(a_change3)
        b_change3_list.append(b_change3)
        a_n_list.append(a_n)
        b_n_list.append(b_n)

        # update ratings using the outcome (if we have it)
        y = df.at[i, y_col] if y_col in df.columns else None
        if pd.isna(y):
            # still update history trackers so "form" evolves smoothly
            last_pre[a_id].append(a_pre)
            last_pre[b_id].append(b_pre)
            continue

        try:
            s_a = float(y)
        except Exception:
            last_pre[a_id].append(a_pre)
            last_pre[b_id].append(b_pre)
            continue

        if s_a not in (0.0, 1.0):
            last_pre[a_id].append(a_pre)
            last_pre[b_id].append(b_pre)
            continue

        e_a = _expected_score(a_pre, b_pre, scale)

        # method/round weighting
        w = 1.0
        if method_col in df.columns or round_col in df.columns:
            m = df.at[i, method_col] if method_col in df.columns else None
            r = df.at[i, round_col] if round_col in df.columns else None
            w *= _mov_weight(m, r)

        # 5-round fights get a slight bump
        if scheduled_rounds_col in df.columns:
            sr = df.at[i, scheduled_rounds_col]
            try:
                if int(sr) == 5:
                    w *= 1.08
            except Exception:
                pass

        delta = k * w * (s_a - e_a)

        ratings[a_id] = a_pre + delta
        ratings[b_id] = b_pre - delta

        fights_count[a_id] += 1
        fights_count[b_id] += 1

        last_pre[a_id].append(a_pre)
        last_pre[b_id].append(b_pre)

    df["a_elo_pre"] = a_pre_list
    df["b_elo_pre"] = b_pre_list
    df["a_elo_change_last_3"] = a_change3_list
    df["b_elo_change_last_3"] = b_change3_list
    df["a_elo_fights"] = a_n_list
    df["b_elo_fights"] = b_n_list

    return df
