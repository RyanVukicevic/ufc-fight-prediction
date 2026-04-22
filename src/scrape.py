

#as of now this file serves to hold all functions aiding and making the scraping of upcoming events 
#and their fights possible

#might be adapted to do past events in future


from datetime import datetime
import requests
from bs4 import BeautifulSoup


BASE_URL = "http://ufcstats.com"

def get_event_date(event_url: str) -> str | None:
    r = requests.get(event_url, timeout=20)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    items = soup.select("li.b-list__box-list-item")
    for item in items:
        title = item.select_one("i.b-list__box-item-title")
        if title and title.get_text(strip=True).lower() == "date:":
            text = item.get_text(" ", strip=True)
            return text.replace("Date:", "", 1).strip()

    return None

def parse_date(date_str: str | None) -> datetime | None:
    "helper for get_upcoming_events(), turns dates -> datetime"
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%B %d, %Y")
    except ValueError:
        return None


def get_upcoming_events() -> list[dict]:
    """
    Returns:
      [{"name": "...", "url": "...", "date_str": "...", "date": datetime}, ...]
    """
    url = f"{BASE_URL}/statistics/events/upcoming"
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
            "date": date_obj
        })

    return events



def get_event_fights(event_url: str) -> list[dict]:
    """
    Returns fights for one event:
      [{"fighter_a": "...", "fighter_b": "..."}, ...]
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

        red = fighters[0].get_text(strip=True)
        blue = fighters[1].get_text(strip=True)

        fights.append({"fighter_a": red, "fighter_b": blue})

    return fights

def get_upcoming_fights_grouped() -> list[dict]:
    """
    Returns events grouped with fights:
      [
        {
          "name": "...",
          "url": "...",
          "date_str": "...",
          "date": datetime,
          "fights": [
            {"fighter_a": "...", "fighter_b": "..."}
          ]
        },
        ...
      ]
    """
    events = get_upcoming_events()
    for e in events:
        e["fights"] = get_event_fights(e["url"])
    return events
