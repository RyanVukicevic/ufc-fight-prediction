
from src.raw_data import load_raw_data
from src.clean_data import load_cleaned_data
from src.x_and_y import make_x_and_y
from src.model import train_logreg, train_boosted, print_metrics
from src.scrape import get_upcoming_fights_grouped
from src.inference import (
    diagnose_unpredictable_fights,
    predict_upcoming_events,
    predictions_to_df,
    print_predictions,
)
from src.odds import get_feature_ranking_by_coef, get_logreg_odds_ratios
from src.cv import cross_validate_logreg, print_cv_summary
from src.interpretations import interpret_cleaning, interpret_cv_and_scores, interpret_odds
from src.plotting import plot_fights_per_fighter
from src.elo import elo_leaderboard
from src.past_events import (
    completed_predictions_to_df,
    get_completed_fights_grouped,
    predict_completed_events,
    summarize_completed_predictions,
)

import pandas as pd

#settings for any viewer to see the full width of any df, as some of them become wide 
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

def printer(df_dict: dict):
    
    for name, df in df_dict.items():
        print(name, df.shape)
        print(df.head(5).to_string())


def main():
    #get raw data, 5 dataframes from a github user's scrape of ufc stats website
    #so data is included, no external download neccessary, fully intergrated into pipeline
    raw = load_raw_data()
    print("STEP 1 RAW DATA\n")
    printer(raw)

    #clean all data and make multiple useful utility dataframes, as shown
    cleaned = load_cleaned_data(raw)
    print("\nSTEP 2 CLEANED\n")
    printer(cleaned)

    interpret_cleaning()

    TRAIN_SIZE = 0.8

    #get x, y and x_train, x_test, y_train, y_test from cleaned["full_sym"]
    print("\nSTEP 3 XY, XY_SPLITS\n")
    xy, xy_splits = make_x_and_y(cleaned["full_sym"], train_size=TRAIN_SIZE)
    printer(xy)
    printer(xy_splits)

    #trained both logistic regression and boosted decision tree models and giving info/metrics
    print("\nSTEP 4 MODELS\n")
    model_logreg, model_logreg_metrics = train_logreg(xy_splits)
    model_boosted, model_boosted_metrics = train_boosted(xy_splits)
    print_metrics("LOGREG", model_logreg, model_logreg_metrics)
    print("\n")
    print_metrics("BOOSTED", model_boosted, model_boosted_metrics)

    #5-fold cv for proving model robustness, regardless of the train/test split 
    print("\nSTEP 4.5 LOGREG CROSS VALIDATION\n")
    cv_results_df, cv_summary = cross_validate_logreg(xy_splits, n_splits=5)
    print_cv_summary(cv_results_df, cv_summary)
    print(cv_results_df.to_string(index=False))

    plot_fights_per_fighter(cleaned["fighter_stats"])

    interpret_cv_and_scores()

    #showing most productive features for determining if a fighter wins/loses
    print("\nSTEP 4.5 ODDS RATIOS\n")
    odds_df = get_logreg_odds_ratios(model_logreg,xy_splits["x_train"].columns)

    print("most influential feats, as larger abs(coefficient) -> more important\n")
    top_abs = get_feature_ranking_by_coef(odds_df)
    print(top_abs.to_string(index=False, formatters={
        "coefficient": "{:.2f}".format,
        "odds_ratio": "{:.2f}".format,
    }))

    interpret_odds()

    print("\nSTEP 4.5 DISPLAY ELO'S ROBUSTNESS WITH FIGHTER LEADERBOARD\n")

    print(elo_leaderboard(cleaned["fighter_stats"], top_n=25).to_string(index=False))

    print("\nMany of the sports' greatest legacy and current fighters are here,\n" \
    "this acts leaderboard acts as a somewhat informal way of 'proving' the competency\n"
    "of my elo system, as it should rank fighters highly based also on the quality of opponent\n" \
    "so the idea being if the GOATs of the sport and current top fighters are here, the system is designed well\n")


    #i scraped my own upcoming event info to get all future fights for inference, 
    #pulled from http://ufcstats.com/statistics/events/upcoming
    print("\nSTEP 5 SCRAPE UPCOMING EVENT INFO\n")
    events_final = get_upcoming_fights_grouped()
    for d in events_final:
        print()
        for k,v in d.items():
            print(f"{k}: {v}")

    #diagnosis for me on fights that couldnt be predicted, mostly because of debuting fighters
    #not being in the database already, nothing i can really do about that

    # diagnostics = diagnose_unpredictable_fights(events_final, cleaned)
    # print(diagnostics.shape)
    # print(diagnostics.to_string(index=False))

    #use scraped info to predict all fights with logreg model as it is interpretable, performs well
    print("\nSTEP 7 PREDICT ALL UPCOMING FIGHTS\n")
    predictions = predict_upcoming_events(events_final, cleaned, model_logreg, xy_splits["x_train"].columns)
    print_predictions(predictions)

    #storing predictions in dataframe for clean access, potential analytics
    print("\nSTEP 8 STORE PREDICTIONS IN DB\n")
    predictions_df = predictions_to_df(predictions)
    print(predictions_df.head(10).to_string(index=False))


    # Run this after main() has created cleaned, model_logreg, and xy_splits,
    # or paste these lines inside main() after STEP 4 if you want it in the full demo flow.
    #
    completed_events = get_completed_fights_grouped(limit=5)

    #predicts past 5 events and returns in dataframe
    print("\nSTEP 9 PREDICT PAST EVENTS\n")
    completed_predictions = predict_completed_events(
        completed_events,
        cleaned,
        model_logreg,
        xy_splits["x_train"].columns,
    )
    completed_df = completed_predictions_to_df(completed_predictions)
    print(completed_df.to_string(index=False))
    print(summarize_completed_predictions(completed_df))




if __name__ == "__main__":
    main()

