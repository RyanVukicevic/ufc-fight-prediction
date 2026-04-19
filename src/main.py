
from src.raw_data import load_raw_data
from src.clean_data import load_cleaned_data
from src.x_and_y import make_x_and_y, train_test_split_xy
from src.model import train_logreg, train_boosted, print_metrics
from src.scrape import *
from src.inference import *

from IPython.display import display
import pandas as pd

def printer(df_dict: dict):
    
    for name, df in df_dict.items():
        print(name, df.shape)
        display(df.head(5))

def main():

    #get raw data, 5 dataframes from github user's scrape of ufc websites
    #so data is included, no external download, fully intergrated into pipeline
    raw = load_raw_data()
    print("STEP 1 RAW DATA\n")
    printer(raw)
    
    #clean all data and make multiple useful utility dataframes, as shown
    cleaned = load_cleaned_data(raw)
    print("\nSTEP 2 CLEANED\n")
    printer(cleaned)

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

    #scraped upcoming event info to get all future fights for inference, 
    #pulled from http://ufcstats.com/statistics/events/upcoming
    print("\nSTEP 5 SCRAPE UPCOMING EVENT INFO\n")
    events_final = get_upcoming_fights_grouped()
    for d in events_final:
        print()
        for k,v in d.items():
            print(f"{k}: {v}")

    #use scraped infor to predict all fights with logreg model as it is interpretable, performs well
    print("\nSTEP 6 PREDICT ALL UPCOMING FIGHTS\n")
    predictions = predict_upcoming_events(events_final, cleaned, model_logreg, xy_splits["x_train"].columns)
    print_predictions(predictions)

    #storing predictions in dataframe for clean access, potential analytics
    print("\nSTEP 7 STORE PREDICTIONS IN DB\n")
    predictions_df = predictions_to_df(predictions)
    display(predictions_df.head(10))


main()
