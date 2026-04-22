# UFC Fight Prediction Project

## Overview
This project predicts UFC fight outcomes using historical UFCStats data, engineered pre-fight features, and machine learning models.

The current pipeline:

- Loads historical UFCStats CSV data from a public GitHub scrape
- Cleans fights and fighter metadata
- Includes both men's and women's UFC fights
- Engineers pre-fight matchup features as fighter-stat deltas
- Adds a chronological Elo rating system
- Trains Logistic Regression and HistGradientBoosting models
- Scrapes upcoming UFC events from UFCStats
- Predicts upcoming fights when both fighters have usable UFC history
- Diagnoses why some fights cannot be predicted
- Scrapes completed events for display/evaluation against known results
- Provides an Elo leaderboard for sanity-checking fighter strength rankings

## How to Run
From the project root:

```bash
pip install -r requirements.txt
python -m src.main
```

The main demo notebook is:

```text
demo.ipynb
```

The notebook is the best way to review intermediate DataFrames, metrics, odds ratios, prediction tables, and visual outputs.

## Data
Historical data is loaded from:

```text
https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/main/
```

Upcoming and completed event cards are scraped directly from:

```text
http://ufcstats.com
```

No local dataset is required.

## Feature Engineering
Each fight is represented from both fighter perspectives. For example, a fight appears once as Fighter A vs Fighter B and once as Fighter B vs Fighter A.

The model receives mostly `delta_*` features:

```text
delta_feature = fighter_A_feature - fighter_B_feature
```

Examples:

- `delta_elo_pre`
- `delta_age_at_fight`
- `delta_wins_entering`
- `delta_losses_entering`
- `delta_win_rate_entering`
- `delta_reach`
- `delta_avg_fight_time_entering`
- `delta_five_round_rate_entering`

Weightclass is kept for cleaning, display, and height/reach imputation, but it is not used as a one-hot model input.

## Elo System
The Elo system is built chronologically so each fight only uses ratings available before that fight.

It adds:

- `elo_pre`: fighter's rating before the fight
- `elo_change_last_3`: recent Elo movement
- `elo_fights`: prior Elo-tracked UFC fights

The update accounts for:

- opponent rating
- win/loss result
- method of victory
- round of finish
- slight five-round fight weighting

The Elo leaderboard can be generated with:

```python
from src.elo import elo_leaderboard

elo_leaderboard(cleaned["fighter_stats"], top_n=25)
```

## Models
The project trains:

- Logistic Regression with median imputation and standard scaling
- HistGradientBoostingClassifier

Logistic Regression is useful for interpretation through coefficients and odds ratios. The boosted model is also evaluated because it can capture nonlinear patterns.

## Inference
Upcoming events are scraped from UFCStats and passed through the same feature-building logic used for training.

The inference code:

- normalizes known fighter-name aliases
- builds latest pre-fight snapshots
- includes Elo deltas
- outputs predicted winner and win probability
- marks fights as unpredicted if fighter history is unavailable

## Missing Prediction Diagnostics
Some upcoming fights cannot be predicted because one or both fighters lack model-usable history.

`diagnose_unpredictable_fights(...)` separates those cases into categories such as:

- `not_in_fighters_table`
- `profile_exists_no_cleaned_fights`
- `no_prior_eligible_snapshot`

This helps distinguish likely debutants from name mismatch/source-data issues.

## Completed Event Evaluation
The project can scrape completed UFCStats events and compare model predictions to actual winners.

Example:

```python
from src.past_events import (
    get_completed_fights_grouped,
    predict_completed_events,
    completed_predictions_to_df,
    summarize_completed_predictions,
)

completed_events = get_completed_fights_grouped(limit=5)
completed_predictions = predict_completed_events(
    completed_events,
    cleaned,
    model_logreg,
    xy_splits["x_train"].columns,
)
completed_df = completed_predictions_to_df(completed_predictions)
summarize_completed_predictions(completed_df)
```

This is display/evaluation mode, not a strict rolling historical backtest.

## Results
Recent notebook results are around:

- Logistic Regression accuracy: about 61%
- Logistic Regression ROC-AUC: about 0.66
- HistGradientBoosting accuracy: about 62%
- HistGradientBoosting ROC-AUC: about 0.66

These numbers are reasonable for UFC prediction because many fights are close, fighters have limited sample sizes, and injuries/matchup dynamics are not fully captured by historical data.

## Project Structure
```text
src/
  raw_data.py              Load raw UFCStats CSV data
  clean_data_with_elo.py   Main cleaning pipeline with Elo features
  clean_data.py            Non-Elo/legacy cleaning pipeline
  elo.py                   Elo rating features and leaderboard
  x_and_y.py               Feature matrix and train/test split
  model.py                 Model training and metrics
  cv.py                    Cross-validation helpers
  odds.py                  Logistic-regression odds ratios
  scrape.py                Upcoming event scraping
  inference.py             Upcoming fight prediction and diagnostics
  past_events.py           Completed event scraping/evaluation
  lookup.py                Fighter-name lookup/debug helpers
  plotting.py              Plotting helpers
  interpretations.py       Printed project interpretations
  main.py                  Script entry point
```

## Notes and Limitations
- A fight may not be predicted if one or both fighters have no prior usable UFC history.
- Fighters with new UFCStats profiles but no completed UFC fights are often likely debutants.
- Name aliases are handled manually for known mismatches.
- The completed-event evaluation uses the current trained model; it is not a full rolling retrain backtest.
- Data freshness depends on the upstream GitHub scrape and UFCStats pages.
