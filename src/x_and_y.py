
import pandas as pd
import numpy as np

def make_x_and_y(full_sym: pd.DataFrame, train_size: int):
    """returns 2 dictionaries
    x_and_y: hold x, y before split
    splits: hold x_train, x_test, y_train, y_test obtained from x, y with specified train_size for split"""

    x, y = get_x_and_y(full_sym)

    x_train, x_test, y_train, y_test = train_test_split_xy(x, y, train_size=train_size)

    x_and_y = {"x":x, "y":y}
    splits = {"x_train":x_train, "x_test":x_test, "y_train":y_train, "y_test": y_test}

    return x_and_y, splits


from IPython.display import display

def get_x_and_y(full_sym: pd.DataFrame):
    """takes full_sym df, sorts by descending date, and keeps deltas + some base cols
    while also engineering x, y"""

    df = full_sym.copy()


    # sort chronologically (stable for same-date fights)
    df = df.sort_values(["date", "bout_url", "fighter_a"], kind="mergesort").reset_index(drop=True)

    # now this adjacency check should work because for the same bout_url,
    # the two mirrored rows are next to each other within the same date
    assert (df["bout_url"].shift(1) == df["bout_url"]).iloc[1::2].all()


    delta_cols = [c for c in df.columns if c.startswith("delta_")]
    base_cols = [c for c in ["date", "ppv", "scheduled_rounds"] if c in df.columns]

    X = df[base_cols + delta_cols].copy()
    y = df["y"].astype(int)

    bool_cols = X.select_dtypes(include=["bool"]).columns
    X[bool_cols] = X[bool_cols].astype(int)

    X = X.drop(columns="date")


    return X, y


def train_test_split_xy(x, y, train_size=0.8):
    """splits x, y into x_train, x_test, y_train, y_test with specified train_size
    train_size defaults to 0.8"""

    # handle y as series
    y = pd.Series(y).reset_index(drop=True)
    x = x.reset_index(drop=True)

    n = len(x)
    assert n == len(y), "X and y must have same length"
    assert n % 2 == 0, "need even number of rows (2 per fight)"

    n_fights = n // 2
    cut = int(n_fights * (train_size))
    split_row = 2 * cut

    x_train = x.iloc[:split_row].reset_index(drop=True)
    y_train = y.iloc[:split_row].reset_index(drop=True)

    x_test  = x.iloc[split_row:].reset_index(drop=True)
    y_test  = y.iloc[split_row:].reset_index(drop=True)

    return x_train, x_test, y_train, y_test
