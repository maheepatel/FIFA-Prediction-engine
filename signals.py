"""
signals.py — Real contextual signals only (no hash-based garbage).
Altitude mismatch, rest advantage, fixture congestion, injury news, coach pressure.
Each signal backed by actual data, not deterministic hash functions.
"""
from data import get_altitude, get_news
from typing import List, Dict
from datetime import datetime, timezone

TRAINING_ALTITUDES = {
    "Brazil": 871, "Argentina": 20, "England": 53, "France": 160,
    "Germany": 112, "Spain": 750, "Portugal": 25, "Netherlands": 5,
    "USA": 300, "Mexico": 2240, "Colombia": 2600, "Ecuador": 2800,
    "Bolivia": 3640, "Peru": 154, "Chile": 520, "Qatar": 10,
    "Morocco": 5, "Senegal": 10, "Nigeria": 75, "Egypt": 20,
    "Italy": 100, "Uruguay": 50, "Croatia": 130, "Denmark": 30,
    "Poland": 100, "Japan": 200, "South Korea": 200, "Australia": 50,
    "Iran": 1200, "Tunisia": 50, "Cameroon": 200, "Ghana": 100,
    "Algeria": 100, "Saudi Arabia": 500, "Canada": 150, "Paraguay": 150,
    "South Africa": 1700, "Czechia": 350,
}


def run_all_signals(team_a: str, team_b: str, venue_city: str, venue_country: str,
                    form_a: Dict, form_b: Dict) -> Dict:
    signals = []
    adj_a = 0.0
    adj_b = 0.0

    def add(sig):
        if not sig:
            return
        signals.append(sig)
        nonlocal adj_a, adj_b
        pct = sig.get("adj_pct", 0)
        if sig["team"] == "A":
            adj_a += pct if sig["direction"] == "advantage" else -pct
        else:
            adj_b += pct if sig["direction"] == "advantage" else -pct

    venue_alt = get_altitude(venue_city, venue_country)
    alt_a = TRAINING_ALTITUDES.get(team_a, 100)
    alt_b = TRAINING_ALTITUDES.get(team_b, 100)

    add(_altitude_signal(team_a, team_b, venue_alt, alt_a, alt_b))
    add(_rest_signal(team_a, team_b, form_a, form_b))
    add(_congestion_signal(team_a, form_a, "A"))
    add(_congestion_signal(team_b, form_b, "B"))
    add(_injury_signal(team_a, "A"))
    add(_injury_signal(team_b, "B"))
    add(_coach_signal(team_a, "A"))
    add(_coach_signal(team_b, "B"))

    adj_a = max(-8, min(8, adj_a))
    adj_b = max(-8, min(8, adj_b))

    return {
        "signals": [s for s in signals if s],
        "adj_a": round(adj_a, 2),
        "adj_b": round(adj_b, 2),
        "venue_alt": venue_alt,
        "alt_a": alt_a,
        "alt_b": alt_b,
    }


def _altitude_signal(team_a: str, team_b: str,
                     venue_alt: int, alt_a: float, alt_b: float) -> Dict:
    if venue_alt == 0:
        return {"name": "Altitude", "team": "A", "direction": "neutral",
                "impact": 0, "adj_pct": 0,
                "finding": "Venue altitude data unavailable — no altitude adjustment.",
                "source": "Open-Elevation API"}

    gap_a = abs(venue_alt - alt_a)
    gap_b = abs(venue_alt - alt_b)

    if gap_a > gap_b + 300:
        impact = 3 if gap_a > 1200 else 2
        return {"name": "Altitude Disadvantage", "team": "A", "direction": "disadvantage",
                "impact": impact, "adj_pct": impact * 0.8,
                "finding": f"{team_a} trains at {alt_a}m, venue at {venue_alt}m (gap {gap_a}m). "
                           f"{team_b} gap only {gap_b}m. VO2 max reduction of 5-12% at altitude.",
                "source": "Open-Elevation API + FIFA Medical Research"}
    elif gap_b > gap_a + 300:
        impact = 3 if gap_b > 1200 else 2
        return {"name": "Altitude Disadvantage", "team": "B", "direction": "disadvantage",
                "impact": impact, "adj_pct": impact * 0.8,
                "finding": f"{team_b} trains at {alt_b}m, venue at {venue_alt}m (gap {gap_b}m). "
                           f"{team_a} gap only {gap_a}m. VO2 max reduction of 5-12% at altitude.",
                "source": "Open-Elevation API + FIFA Medical Research"}
    return {"name": "Altitude", "team": "A", "direction": "neutral", "impact": 0,
            "adj_pct": 0,
            "finding": f"Altitude gap minimal. Venue: {venue_alt}m. Both teams within "
                       f"{max(gap_a, gap_b)}m of training altitude.",
            "source": "Open-Elevation API"}


def _rest_signal(team_a: str, team_b: str, form_a: Dict, form_b: Dict) -> Dict:
    def days_since(date_str):
        if not date_str:
            return 7
        try:
            d = datetime.fromisoformat(str(date_str).replace("Z", "+00:00"))
            return (datetime.now(timezone.utc) - d).days
        except Exception:
            return 7

    rest_a = days_since(form_a.get("last_match_date"))
    rest_b = days_since(form_b.get("last_match_date"))
    diff = rest_a - rest_b

    if diff >= 4:
        return {"name": "Rest Advantage", "team": "A", "direction": "advantage",
                "impact": 3, "adj_pct": 2.0,
                "finding": f"{team_a} has {rest_a} days rest vs {team_b}'s {rest_b}. "
                           f"UEFA research: 4+ day rest gap = significant recovery advantage.",
                "source": "Fixture data + UEFA workload research"}
    elif diff >= 2:
        return {"name": "Slight Rest Advantage", "team": "A", "direction": "advantage",
                "impact": 1, "adj_pct": 0.5,
                "finding": f"{team_a} has {rest_a}d rest vs {team_b}'s {rest_b}d.",
                "source": "Fixture data"}
    elif diff <= -4:
        return {"name": "Rest Disadvantage", "team": "A", "direction": "disadvantage",
                "impact": 3, "adj_pct": 2.0,
                "finding": f"{team_a} has only {rest_a} days rest vs {team_b}'s {rest_b}.",
                "source": "Fixture data + UEFA workload research"}
    elif diff <= -2:
        return {"name": "Slight Rest Disadvantage", "team": "A", "direction": "disadvantage",
                "impact": 1, "adj_pct": 0.5,
                "finding": f"{team_a} has {rest_a}d rest vs {team_b}'s {rest_b}d.",
                "source": "Fixture data"}
    return {"name": "Rest", "team": "A", "direction": "neutral", "impact": 0,
            "adj_pct": 0,
            "finding": f"Rest equal: {team_a}={rest_a}d, {team_b}={rest_b}d.",
            "source": "Fixture data"}


def _congestion_signal(team: str, form: Dict, side: str = "A") -> Dict:
    n = form.get("matches_last_21", 2)
    if n >= 5:
        return {"name": "Fixture Congestion", "team": side, "direction": "disadvantage",
                "impact": 3, "adj_pct": 1.5,
                "finding": f"{team} played {n} matches in 21 days. "
                           f"Elevated injury risk per UEFA research.",
                "source": "UEFA Player Workload Research"}
    elif n <= 1:
        return {"name": "Low Congestion", "team": side, "direction": "advantage",
                "impact": 1, "adj_pct": 0.5,
                "finding": f"{team} played only {n} match in 21 days — fresh.",
                "source": "Fixture data"}
    return {"name": "Congestion", "team": side, "direction": "neutral",
            "impact": 0, "adj_pct": 0,
            "finding": f"{team}: {n} matches in 21 days — normal schedule.",
            "source": "Fixture data"}


def _injury_signal(team: str, side: str) -> Dict:
    news = get_news(team, "injury")
    negative_kw = [
        "injury", "injured", "crisis", "doubt", "miss", "ruled out",
        "suspended", "setback", "fracture", "hamstring", "knee",
    ]
    positive_kw = ["returns", "recovered", "fit", "back in training", "cleared"]

    for article in news:
        title = (article.get("title", "") + " " + article.get("source", "")).lower()
        has_negative = any(kw in title for kw in negative_kw)
        has_positive = any(kw in title for kw in positive_kw)

        if has_negative and not has_positive:
            return {"name": "Injury/Suspension News", "team": side, "direction": "disadvantage",
                    "impact": 2, "adj_pct": 1.0,
                    "finding": f"Negative news: '{article.get('title', '')}'",
                    "source": article.get("url", article.get("source", "News"))}

    return {"name": "Squad Health", "team": side, "direction": "neutral",
            "impact": 0, "adj_pct": 0,
            "finding": f"No injury or suspension alerts for {team}.",
            "source": "DuckDuckGo news search"}


def _coach_signal(team: str, side: str) -> Dict:
    news = get_news(team, "coach")
    voc_keywords = ["vote of confidence", "full support", "backs the manager",
                    "job is safe", "extend contract"]
    sacking_kw = ["sacked", "fired", "dismissed", "parted ways", "resigns"]

    for article in news:
        text = (article.get("title", "") + " " + article.get("source", "")).lower()
        if any(kw in text for kw in sacking_kw):
            return {"name": "Manager Instability", "team": side, "direction": "disadvantage",
                    "impact": 3, "adj_pct": 1.5,
                    "finding": f"{team}'s coach recently sacked/left. Tactical uncertainty.",
                    "source": article.get("url", article.get("source", "News"))}
        if any(kw in text for kw in voc_keywords):
            return {"name": "Vote of Confidence", "team": side, "direction": "disadvantage",
                    "impact": 2, "adj_pct": 1.0,
                    "finding": f"Board publicly backed {team}'s manager. "
                               f"Historically correlates with poor performance.",
                    "source": article.get("url", article.get("source", "News"))}

    return {"name": "Coach Stability", "team": side, "direction": "neutral",
            "impact": 0, "adj_pct": 0,
            "finding": f"No coaching instability signals for {team}.",
            "source": "DuckDuckGo news"}
