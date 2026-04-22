import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.inference import normalize_fighter_name, predict_one_fight
from src.scrape import BASE_URL, get_event_date, parse_date


def get_completed_events(limit=None) -> list[dict]:
    """
    Scrape completed UFCStats events.

    Args:
        limit: optional number of newest completed events to return.
    """
    url = f"{BASE_URL}/statistics/events/completed?page=all"
    r = requests.get(url, timeout=20)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    events: list[dict] = []

    rows = soup.select("tr.b-statistics__table-row")
    for row in rows:
        link = row.select_one("a[href*='event-details']")
        if not link:
            continue

        name = link.get_text(strip=True)
        event_url = link["href"]

        date_str = get_event_date(event_url)
        date_obj = parse_date(date_str)

        events.append({
            "name": name,
            "url": event_url,
            "date_str": date_str,
            "date": date_obj,
        })

        if limit is not None and len(events) >= limit:
            break

    return events


def get_completed_event_fights(event_url: str) -> list[dict]:
    """
    Returns completed fights for one event.

    Each row includes names and the UFCStats fight-details URL when available.
    """
    r = requests.get(event_url, timeout=20)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    fights: list[dict] = []

    rows = soup.select("tbody.b-fight-details__table-body tr")
    for row in rows:
        fighters = row.select("a.b-link.b-link_style_black[href*='fighter-details']")
        if len(fighters) != 2:
            continue

        fighter_a = fighters[0].get_text(strip=True)
        fighter_b = fighters[1].get_text(strip=True)

        bout_url = row.get("data-link")
        if not bout_url:
            bout_link = row.select_one("a[href*='fight-details']")
            bout_url = bout_link["href"] if bout_link else None

        fights.append({
            "fighter_a": fighter_a,
            "fighter_b": fighter_b,
            "bout_url": bout_url,
        })

    return fights


def get_completed_fights_grouped(limit=None) -> list[dict]:
    """
    Returns completed events grouped with fight lists.
    """
    events = get_completed_events(limit=limit)
    for event in events:
        event["fights"] = get_completed_event_fights(event["url"])
    return events


def _normalized_name_set(*names):
    return {normalize_fighter_name(name) for name in names if pd.notna(name)}


def find_actual_fight_row(fight: dict, event: dict, full_fight_stats: pd.DataFrame):
    """
    Find the cleaned fight row corresponding to a scraped completed fight.

    Prefer bout_url because it is stable, then fall back to same date plus
    unordered fighter-name matching.
    """
    bout_url = fight.get("bout_url")
    if bout_url:
        by_url = full_fight_stats[full_fight_stats["bout_url"] == bout_url]
        if not by_url.empty:
            return by_url.iloc[0]

    event_date = pd.to_datetime(event["date"])
    target_names = _normalized_name_set(fight["fighter_a"], fight["fighter_b"])

    same_date = full_fight_stats[pd.to_datetime(full_fight_stats["date"]) == event_date]
    for _, row in same_date.iterrows():
        row_names = _normalized_name_set(row["fighter_a"], row["fighter_b"])
        if row_names == target_names:
            return row

    return None


def predict_completed_events(events, cleaned, model, x_columns) -> list[dict]:
    """
    Predict completed events and attach actual winners from cleaned history.

    This is display/evaluation mode: it uses the already-trained model and
    pre-fight snapshots, then compares predictions against known results.
    """
    all_events = []
    full_fight_stats = cleaned["full_fight_stats"]

    for event in events:
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
                x_columns,
            )

            actual_row = find_actual_fight_row(fight, event, full_fight_stats)
            actual_winner = None if actual_row is None else actual_row["winner"]
            actual_loser = None if actual_row is None else actual_row["loser"]

            if pred is None:
                pred = {
                    "fighter_a": fighter_a,
                    "fighter_b": fighter_b,
                    "fight_date": event_date,
                    "event_name": event_name,
                    "fight_idx": fight_idx,
                    "prob_fighter_a_wins": None,
                    "prob_fighter_b_wins": None,
                    "predicted_winner": None,
                }

            pred["bout_url"] = fight.get("bout_url")
            pred["actual_winner"] = actual_winner
            pred["actual_loser"] = actual_loser
            pred["could_predict"] = pred["predicted_winner"] is not None
            pred["actual_found"] = actual_winner is not None
            pred["correct"] = (
                None if not pred["could_predict"] or not pred["actual_found"]
                else normalize_fighter_name(pred["predicted_winner"]) == normalize_fighter_name(actual_winner)
            )

            event_predictions.append(pred)

        all_events.append({
            "event": event_name,
            "date": event_date,
            "predictions": event_predictions,
        })

    return all_events


def completed_predictions_to_df(predictions) -> pd.DataFrame:
    rows = []

    for event in predictions:
        for pred in event["predictions"]:
            rows.append({
                "event": event["event"],
                "date": event["date"],
                "fight_idx": pred["fight_idx"],
                "fighter_a": pred["fighter_a"],
                "fighter_b": pred["fighter_b"],
                "bout_url": pred["bout_url"],
                "prob_fighter_a_wins": pred["prob_fighter_a_wins"],
                "prob_fighter_b_wins": pred["prob_fighter_b_wins"],
                "predicted_winner": pred["predicted_winner"],
                "actual_winner": pred["actual_winner"],
                "actual_loser": pred["actual_loser"],
                "could_predict": pred["could_predict"],
                "actual_found": pred["actual_found"],
                "correct": pred["correct"],
            })

    return pd.DataFrame(rows)


def summarize_completed_predictions(predictions_df: pd.DataFrame) -> dict:
    predicted = predictions_df[predictions_df["could_predict"] & predictions_df["actual_found"]].copy()

    return {
        "total_fights": len(predictions_df),
        "predicted_fights": int(predictions_df["could_predict"].sum()),
        "actuals_found": int(predictions_df["actual_found"].sum()),
        "evaluated_fights": len(predicted),
        "skipped_fights": int((~predictions_df["could_predict"]).sum()),
        "accuracy": None if predicted.empty else float(predicted["correct"].mean()),
    }
