

#this file stores all of the functions used in inference, so when i scrape upcoming cards and 
#extract all upcoming fights, the functions here transform the scraped names by matching them to those
#in the existing database (collection of dataframes), compute their stats and therefore deltas (difference)
#all for prediction

import pandas as pd


NAME_ALIASES = {
    "Waldo Cortes Acosta": "Waldo Cortes-Acosta",
    "Zach Reese": "Zachary Reese",
    "Shem Rock": "Shaqueme Rock",
    "Michael Aswell Jr.": "Michael Aswell",
    "Rafael Cerquiera": "Rafael Cerqueira",
    "Benjamin Johnston": "Benjamin Donovan Johnston",
}


def normalize_fighter_name(fighter_name):
    return NAME_ALIASES.get(fighter_name, fighter_name)


def get_fighter_url(fighter_name, fighters):
    fighter_name = normalize_fighter_name(fighter_name)
    name_to_url = fighters.set_index("fighter")["fighter_url"].to_dict()
    return name_to_url.get(fighter_name)


def fighter_has_any_snapshot(fighter_name, fighters, fighter_stats):
    fighter_url = get_fighter_url(fighter_name, fighters)
    if fighter_url is None:
        return False
    return fighter_stats["fighter_url"].eq(fighter_url).any()


def get_latest_snapshot(fighter_name, fight_date, fighters, fighter_stats):
    fighter_url = get_fighter_url(fighter_name, fighters)

    if fighter_url is None:
        return None

    rows = fighter_stats[
        (fighter_stats["fighter_url"] == fighter_url) &
        (fighter_stats["date"] < fight_date)
    ].sort_values(["date", "bout_url"])

    if rows.empty:
        return None

    return rows.iloc[-1].copy()


def get_fighter_dob(fighter_name, fighters):
    fighter_name = normalize_fighter_name(fighter_name)
    row = fighters[fighters["fighter"] == fighter_name]
    if row.empty:
        return None

    dob = row.iloc[0]["dob"]
    if pd.isna(dob):
        return None

    return pd.to_datetime(dob)


def compute_age_on_date(dob, fight_date, default_age=30):
    if dob is None or pd.isna(dob):
        return default_age

    dob = pd.to_datetime(dob)
    fight_date = pd.to_datetime(fight_date)

    age = fight_date.year - dob.year
    if (fight_date.month, fight_date.day) < (dob.month, dob.day):
        age -= 1

    return age


def infer_ppv(event_name):
    return int("fight night" not in event_name.lower())


def infer_scheduled_rounds(fight_idx):
    # assume first listed fight is main event
    return 5 if fight_idx == 0 else 3


def add_optional_delta(row, feature_name, a, b):
    if feature_name in a.index and feature_name in b.index:
        row[f"delta_{feature_name}"] = a[feature_name] - b[feature_name]


def build_upcoming_feature_row(
    fighter_a,
    fighter_b,
    fight_date,
    event_name,
    fight_idx,
    fighters,
    fighter_stats
):
    fighter_a = normalize_fighter_name(fighter_a)
    fighter_b = normalize_fighter_name(fighter_b)

    a = get_latest_snapshot(fighter_a, fight_date, fighters, fighter_stats)
    b = get_latest_snapshot(fighter_b, fight_date, fighters, fighter_stats)

    if a is None or b is None:
        return None

    a_days_since = (fight_date - a["date"]).days
    b_days_since = (fight_date - b["date"]).days

    a_dob = get_fighter_dob(fighter_a, fighters)
    b_dob = get_fighter_dob(fighter_b, fighters)

    a_age = compute_age_on_date(a_dob, fight_date)
    b_age = compute_age_on_date(b_dob, fight_date)

    row = {
        "ppv": infer_ppv(event_name),
        "scheduled_rounds": infer_scheduled_rounds(fight_idx),
        "weightclass": a["weightclass"],

        "delta_age_at_fight": a_age - b_age,
        "delta_avg_fight_time_entering": a["avg_fight_time_entering"] - b["avg_fight_time_entering"],
        "delta_days_since_last_fight": a_days_since - b_days_since,
        "delta_fights_entering": a["fights_entering"] - b["fights_entering"],
        "delta_finish_losses_entering": a["finish_losses_entering"] - b["finish_losses_entering"],
        "delta_finish_win_rate_entering": a["finish_win_rate_entering"] - b["finish_win_rate_entering"],
        "delta_finish_wins_entering": a["finish_wins_entering"] - b["finish_wins_entering"],
        "delta_five_round_fights_entering": a["five_round_fights_entering"] - b["five_round_fights_entering"],
        "delta_five_round_rate_entering": a["five_round_rate_entering"] - b["five_round_rate_entering"],
        "delta_height": a["height"] - b["height"],
        "delta_ko_losses_entering": a["ko_losses_entering"] - b["ko_losses_entering"],
        "delta_ko_win_rate_entering": a["ko_win_rate_entering"] - b["ko_win_rate_entering"],
        "delta_ko_wins_entering": a["ko_wins_entering"] - b["ko_wins_entering"],
        "delta_losses_entering": a["losses_entering"] - b["losses_entering"],
        "delta_reach": a["reach"] - b["reach"],
        "delta_sub_losses_entering": a["sub_losses_entering"] - b["sub_losses_entering"],
        "delta_sub_win_rate_entering": a["sub_win_rate_entering"] - b["sub_win_rate_entering"],
        "delta_sub_wins_entering": a["sub_wins_entering"] - b["sub_wins_entering"],
        "delta_win_rate_entering": a["win_rate_entering"] - b["win_rate_entering"],
        "delta_wins_entering": a["wins_entering"] - b["wins_entering"],
    }

    for feature_name in ["elo_pre", "elo_change_last_3", "elo_fights"]:
        add_optional_delta(row, feature_name, a, b)

    return row


def make_inference_matrix(row, x_columns):
    x_one = pd.DataFrame([row])
    if "weightclass" in x_one.columns and any(c.startswith("weightclass_") for c in x_columns):
        x_one = pd.get_dummies(x_one, columns=["weightclass"], drop_first=True)
    else:
        x_one = x_one.drop(columns=["weightclass"], errors="ignore")
    x_one = x_one.reindex(columns=x_columns, fill_value=0)
    return x_one


def predict_one_fight(
    fighter_a,
    fighter_b,
    fight_date,
    event_name,
    fight_idx,
    cleaned,
    model,
    x_columns
):
    fighter_a_model_name = normalize_fighter_name(fighter_a)
    fighter_b_model_name = normalize_fighter_name(fighter_b)

    row = build_upcoming_feature_row(
        fighter_a_model_name,
        fighter_b_model_name,
        fight_date,
        event_name,
        fight_idx,
        cleaned["fighters"],
        cleaned["fighter_stats"]
    )

    if row is None:
        return None

    x_one = make_inference_matrix(row, x_columns)
    proba_a = model.predict_proba(x_one)[0, 1]

    return {
        "fighter_a": fighter_a,
        "fighter_b": fighter_b,
        "fight_date": fight_date,
        "event_name": event_name,
        "fight_idx": fight_idx,
        "prob_fighter_a_wins": proba_a,
        "prob_fighter_b_wins": 1 - proba_a,
        "predicted_winner": fighter_a if proba_a >= 0.5 else fighter_b
    }


def predict_upcoming_events(events_final, cleaned, model, x_columns):
    all_predictions = []

    for event in events_final:
        event_name = event["name"]
        event_date = event["date"]

        event_predictions = []

        for fight_idx, fight in enumerate(event["fights"]):
            fighter_a = fight["fighter_a"]
            fighter_b = fight["fighter_b"]

            pred = predict_one_fight(
                fighter_a,
                fighter_b,
                event_date,
                event_name,
                fight_idx,
                cleaned,
                model,
                x_columns
            )

            if pred is None:
                event_predictions.append({
                    "fighter_a": fighter_a,
                    "fighter_b": fighter_b,
                    "fight_date": event_date,
                    "event_name": event_name,
                    "fight_idx": fight_idx,
                    "prob_fighter_a_wins": None,
                    "prob_fighter_b_wins": None,
                    "predicted_winner": None
                })
            else:
                event_predictions.append(pred)

        all_predictions.append({
            "event": event_name,
            "date": event_date,
            "predictions": event_predictions
        })

    return all_predictions


def print_predictions(predictions):
    for event in predictions:
        print()
        print(f"{event['event']} ({event['date'].date()})")

        for pred in event["predictions"]:
            fighter_a = pred["fighter_a"]
            fighter_b = pred["fighter_b"]

            if pred["prob_fighter_a_wins"] is None:
                print(f"{fighter_a} vs. {fighter_b} -> could not predict")
                continue

            proba = pred["prob_fighter_a_wins"]

            if proba >= 0.5:
                winner = fighter_a
                win_prob = proba
            else:
                winner = fighter_b
                win_prob = 1 - proba

            print(f"{fighter_a} vs. {fighter_b} -> {winner} ({win_prob:.3f})")


def predictions_to_df(predictions):
    rows = []

    for event in predictions:
        for pred in event["predictions"]:
            rows.append({
                "event": event["event"],
                "date": event["date"],
                "fighter_a": pred["fighter_a"],
                "fighter_b": pred["fighter_b"],
                "prob_fighter_a_wins": pred["prob_fighter_a_wins"],
                "prob_fighter_b_wins": pred["prob_fighter_b_wins"],
                "predicted_winner": pred["predicted_winner"],
            })

    return pd.DataFrame(rows)


def diagnose_unpredictable_fights(events_final, cleaned):
    rows = []
    fighters = cleaned["fighters"]
    fighter_stats = cleaned["fighter_stats"]

    for event in events_final:
        event_name = event["name"]
        event_date = event["date"]

        for fight_idx, fight in enumerate(event["fights"]):
            fighter_a = fight["fighter_a"]
            fighter_b = fight["fighter_b"]
            fighter_a_model_name = normalize_fighter_name(fighter_a)
            fighter_b_model_name = normalize_fighter_name(fighter_b)

            a_url = get_fighter_url(fighter_a_model_name, fighters)
            b_url = get_fighter_url(fighter_b_model_name, fighters)
            a_has_any_snapshot = fighter_has_any_snapshot(fighter_a_model_name, fighters, fighter_stats)
            b_has_any_snapshot = fighter_has_any_snapshot(fighter_b_model_name, fighters, fighter_stats)
            a_no_model_history = a_url is not None and not a_has_any_snapshot
            b_no_model_history = b_url is not None and not b_has_any_snapshot

            a_snapshot = (
                None if a_url is None
                else get_latest_snapshot(fighter_a_model_name, event_date, fighters, fighter_stats)
            )
            b_snapshot = (
                None if b_url is None
                else get_latest_snapshot(fighter_b_model_name, event_date, fighters, fighter_stats)
            )

            if a_snapshot is not None and b_snapshot is not None:
                continue

            missing = []
            a_issue = None
            b_issue = None

            if a_url is None:
                a_issue = "not_in_fighters_table"
                missing.append(f"{fighter_a}: not in fighters table")
            elif a_no_model_history:
                a_issue = "profile_exists_no_cleaned_fights"
                missing.append(f"{fighter_a}: profile exists but has no eligible cleaned fight history")
            elif a_snapshot is None:
                a_issue = "no_prior_eligible_snapshot"
                missing.append(f"{fighter_a}: no prior eligible fight snapshot")

            if b_url is None:
                b_issue = "not_in_fighters_table"
                missing.append(f"{fighter_b}: not in fighters table")
            elif b_no_model_history:
                b_issue = "profile_exists_no_cleaned_fights"
                missing.append(f"{fighter_b}: profile exists but has no eligible cleaned fight history")
            elif b_snapshot is None:
                b_issue = "no_prior_eligible_snapshot"
                missing.append(f"{fighter_b}: no prior eligible fight snapshot")

            needs_name_or_data_fix = a_url is None or b_url is None
            has_no_model_history = a_no_model_history or b_no_model_history
            if needs_name_or_data_fix:
                action_needed = "check_alias_or_update_fighter_table"
            elif has_no_model_history:
                action_needed = "profile_exists_no_model_history"
            else:
                action_needed = "check_filtered_or_date_ordered_history"

            generic_issues = sorted({
                issue for issue in [a_issue, b_issue]
                if issue is not None
            })
            generic_reason = "; ".join(generic_issues)

            rows.append({
                "event": event_name,
                "date": event_date,
                "fight_idx": fight_idx,
                "fighter_a": fighter_a,
                "fighter_b": fighter_b,
                "fighter_a_model_name": fighter_a_model_name,
                "fighter_b_model_name": fighter_b_model_name,
                "fighter_a_alias_applied": fighter_a != fighter_a_model_name,
                "fighter_b_alias_applied": fighter_b != fighter_b_model_name,
                "fighter_a_in_fighters": a_url is not None,
                "fighter_b_in_fighters": b_url is not None,
                "fighter_a_has_any_snapshot": a_has_any_snapshot,
                "fighter_b_has_any_snapshot": b_has_any_snapshot,
                "fighter_a_no_model_history": a_no_model_history,
                "fighter_b_no_model_history": b_no_model_history,
                "fighter_a_has_snapshot": a_snapshot is not None,
                "fighter_b_has_snapshot": b_snapshot is not None,
                "fighter_a_issue": a_issue,
                "fighter_b_issue": b_issue,
                "generic_reason": generic_reason,
                "action_needed": action_needed,
                "reason": "; ".join(missing),
            })

    return pd.DataFrame(rows)
