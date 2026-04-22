

#importing data from a certain github user's scrape of ufc stats website

import pandas as pd
import numpy as np

def load_raw_data():
    """returns 5 dataframes in a dictionary, where the keys are the names and the values are the dataframes
    note: it makes all columns lowercase, for convention"""
    
    base = "https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/main/"


    raw =  {
        "event_details": pd.read_csv(base + "ufc_event_details.csv"),
        "fight_details": pd.read_csv(base + "ufc_fight_details.csv"),
        "results": pd.read_csv(base + "ufc_fight_results.csv"),
        "fighters": pd.read_csv(base + "ufc_fighter_details.csv"),
        "tott": pd.read_csv(base + "ufc_fighter_tott.csv"),
    }

    #lowercase on all colnames in data and strip whitespace in cells
    for k in raw:
        raw[k].columns = raw[k].columns.str.lower()
        raw[k] = raw[k].apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    return raw


