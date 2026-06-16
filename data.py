"""
data.py — All external API calls.
Team-aware fallbacks so model differentiates even without API keys.
Polymarket/Kalshi style price fetching as comparison benchmark.
"""
import os
import requests
import json
from typing import Dict, List, Optional
from datetime import datetime, timezone, timedelta
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

FOOTBALL_KEY = os.getenv("FOOTBALL_DATA_KEY", "")
ODDS_KEY = os.getenv("ODDS_API_KEY", "")
APIFOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY", "")

TEAM_STRENGTH_TIERS = {
    "Argentina": 9.5, "Brazil": 9.2, "France": 9.1, "England": 8.9,
    "Spain": 8.8, "Portugal": 8.7, "Germany": 8.5, "Netherlands": 8.4,
    "Italy": 8.3, "Croatia": 8.2, "Uruguay": 8.1, "Belgium": 8.0,
    "Colombia": 8.0, "Denmark": 7.9, "Morocco": 7.8, "Japan": 7.7,
    "Poland": 7.6, "USA": 7.5, "Mexico": 7.5, "Senegal": 7.4,
    "South Korea": 7.3, "Australia": 7.2, "Egypt": 7.2, "Nigeria": 7.3,
    "Ecuador": 7.3, "Algeria": 7.1, "Iran": 7.0, "Tunisia": 7.0,
    "Cameroon": 6.9, "Saudi Arabia": 6.7, "Ghana": 6.8, "Qatar": 6.5,
    "Canada": 7.1, "Bosnia & Herzegovina": 6.8, "Paraguay": 7.0,
    "South Africa": 6.6, "Czechia": 7.4,
}


def _team_aware_goals_scored(team: str) -> float:
    tier = TEAM_STRENGTH_TIERS.get(team, 7.0)
    return round(1.0 + (tier - 5.0) * 0.25, 2)


def _team_aware_goals_conceded(team: str) -> float:
    tier = TEAM_STRENGTH_TIERS.get(team, 7.0)
    return round(1.8 - (tier - 5.0) * 0.15, 2)


def _team_aware_form_str(team: str) -> str:
    tier = TEAM_STRENGTH_TIERS.get(team, 7.0)
    if tier >= 8.5:
        return "WWWDW"
    elif tier >= 7.5:
        return "WWDWL"
    elif tier >= 6.5:
        return "WDWLL"
    else:
        return "LDWLL"


def get_team_form(team_name: str) -> Dict:
    if not FOOTBALL_KEY:
        return _fallback_form(team_name)

    headers = {"X-Auth-Token": FOOTBALL_KEY}
    try:
        r = requests.get(
            "https://api.football-data.org/v4/teams",
            headers=headers, params={"name": team_name}, timeout=8
        )
        teams = r.json().get("teams", [])
        if not teams:
            logger.warning(f"Team not found: {team_name}")
            return _fallback_form(team_name)
        team_id = teams[0]["id"]
    except Exception as e:
        logger.warning(f"Team ID fetch failed: {e}")
        return _fallback_form(team_name)

    try:
        r = requests.get(
            f"https://api.football-data.org/v4/teams/{team_id}/matches",
            headers=headers,
            params={"status": "FINISHED", "limit": 20},
            timeout=8,
        )
        matches = r.json().get("matches", [])
    except Exception as e:
        logger.warning(f"Matches fetch failed: {e}")
        return _fallback_form(team_name)

    return _parse_form(matches, team_name, team_id)


def _parse_form(matches: list, team_name: str, team_id: int) -> Dict:
    form, gs, gc = [], [], []
    cutoff_21 = datetime.now(timezone.utc) - timedelta(days=21)
    recent_count = 0

    for m in matches:
        s = m.get("score", {}).get("fullTime", {})
        hg = s.get("home", 0) or 0
        ag = s.get("away", 0) or 0
        is_home = m["homeTeam"]["id"] == team_id
        gf = hg if is_home else ag
        ga = ag if is_home else hg

        gs.append(gf)
        gc.append(ga)
        form.append("W" if gf > ga else "D" if gf == ga else "L")

        try:
            md = datetime.fromisoformat(m["utcDate"].replace("Z", ""))
            if md >= cutoff_21:
                recent_count += 1
        except Exception:
            pass

    n5 = max(min(5, len(form)), 1)
    n10 = max(min(10, len(form)), 1)
    return {
        "form": form,
        "form_str": "".join(form[:10]),
        "goals_scored_5": round(sum(gs[:n5]) / n5, 2),
        "goals_conceded_5": round(sum(gc[:n5]) / n5, 2),
        "goals_scored_10": round(sum(gs[:n10]) / n10, 2),
        "goals_conceded_10": round(sum(gc[:n10]) / n10, 2),
        "xg_scored_5": round(sum(gs[:n5]) / n5, 2),
        "xg_conceded_5": round(sum(gc[:n5]) / n5, 2),
        "matches_last_21": recent_count,
        "wins_5": form[:5].count("W"),
        "draws_5": form[:5].count("D"),
        "losses_5": form[:5].count("L"),
        "last_match_date": matches[0].get("utcDate") if matches else None,
    }


def _fallback_form(team_name: str) -> Dict:
    gs5 = _team_aware_goals_scored(team_name)
    gc5 = _team_aware_goals_conceded(team_name)
    fstr = _team_aware_form_str(team_name)
    w = fstr[:5].count("W")
    d = fstr[:5].count("D")
    l_ = fstr[:5].count("L")
    return {
        "form": list(fstr),
        "form_str": fstr,
        "goals_scored_5": gs5,
        "goals_conceded_5": gc5,
        "goals_scored_10": round((gs5 + 1.4) / 2, 2),
        "goals_conceded_10": round((gc5 + 1.3) / 2, 2),
        "xg_scored_5": gs5,
        "xg_conceded_5": gc5,
        "matches_last_21": 3,
        "wins_5": w,
        "draws_5": d,
        "losses_5": l_,
        "last_match_date": (datetime.now(timezone.utc) - timedelta(days=7)).isoformat(),
    }


def get_h2h(team_a: str, team_b: str) -> Dict:
    if not FOOTBALL_KEY:
        return _fallback_h2h(team_a, team_b)

    headers = {"X-Auth-Token": FOOTBALL_KEY}
    try:
        id_a = _get_team_id(team_a, headers)
        id_b = _get_team_id(team_b, headers)
        if not id_a or not id_b:
            return _fallback_h2h(team_a, team_b)

        r = requests.get(
            f"https://api.football-data.org/v4/teams/{id_a}/matches",
            headers=headers,
            params={"status": "FINISHED", "limit": 50},
            timeout=8,
        )
        matches = r.json().get("matches", [])
        h2h = [
            m for m in matches
            if m["homeTeam"]["id"] == id_b or m["awayTeam"]["id"] == id_b
        ][:6]

        a_w = d = b_w = 0
        a_g = b_g = 0
        for m in h2h:
            s = m["score"]["fullTime"]
            hg, ag = s.get("home", 0) or 0, s.get("away", 0) or 0
            is_a_home = m["homeTeam"]["id"] == id_a
            gf = hg if is_a_home else ag
            ga = ag if is_a_home else hg
            a_g += gf
            b_g += ga
            if gf > ga:
                a_w += 1
            elif gf == ga:
                d += 1
            else:
                b_w += 1

        n = max(len(h2h), 1)
        return {
            "a_wins": a_w, "draws": d, "b_wins": b_w, "total": n,
            "a_goals_avg": round(a_g / n, 2),
            "b_goals_avg": round(b_g / n, 2),
        }
    except Exception as e:
        logger.warning(f"H2H failed: {e}")
        return _fallback_h2h(team_a, team_b)


def _fallback_h2h(team_a: str, team_b: str) -> Dict:
    tier_a = TEAM_STRENGTH_TIERS.get(team_a, 7.0)
    tier_b = TEAM_STRENGTH_TIERS.get(team_b, 7.0)
    diff = tier_a - tier_b
    if diff > 1.5:
        return {"a_wins": 4, "draws": 1, "b_wins": 1, "total": 6,
                "a_goals_avg": 1.8, "b_goals_avg": 0.7}
    elif diff < -1.5:
        return {"a_wins": 1, "draws": 1, "b_wins": 4, "total": 6,
                "a_goals_avg": 0.7, "b_goals_avg": 1.8}
    elif diff > 0.5:
        return {"a_wins": 3, "draws": 2, "b_wins": 1, "total": 6,
                "a_goals_avg": 1.4, "b_goals_avg": 0.9}
    elif diff < -0.5:
        return {"a_wins": 1, "draws": 2, "b_wins": 3, "total": 6,
                "a_goals_avg": 0.9, "b_goals_avg": 1.4}
    else:
        return {"a_wins": 2, "draws": 2, "b_wins": 2, "total": 6,
                "a_goals_avg": 1.2, "b_goals_avg": 1.2}


def _get_team_id(team_name: str, headers: dict) -> Optional[int]:
    try:
        r = requests.get(
            "https://api.football-data.org/v4/teams",
            headers=headers, params={"name": team_name}, timeout=8
        )
        teams = r.json().get("teams", [])
        return teams[0]["id"] if teams else None
    except Exception:
        return None


def get_odds(team_a: str, team_b: str, competition: str = "soccer_fifa_world_cup") -> Dict:
    if not ODDS_KEY:
        return _fallback_odds(team_a, team_b)

    sport_map = {
        "FIFA World Cup 2026": "soccer_fifa_world_cup",
        "FIFA World Cup": "soccer_fifa_world_cup",
        "UEFA Champions League": "soccer_uefa_champs_league",
        "Champions League": "soccer_uefa_champs_league",
        "Premier League": "soccer_epl",
        "Copa America": "soccer_conmebol_copa_america",
        "UEFA Euro": "soccer_european_championship",
        "Other": "soccer_fifa_world_cup",
        "default": "soccer_fifa_world_cup",
    }
    sport = sport_map.get(competition, sport_map["default"])

    try:
        r = requests.get(
            f"https://api.the-odds-api.com/v4/sports/{sport}/odds",
            params={
                "apiKey": ODDS_KEY,
                "regions": "eu",
                "markets": "h2h",
                "oddsFormat": "decimal",
                "bookmakers": "pinnacle,bet365",
            },
            timeout=10,
        )
        games = r.json()
        for game in games:
            home = game.get("home_team", "").lower()
            away = game.get("away_team", "").lower()
            if (team_a.lower() in home or team_a.lower() in away) and \
               (team_b.lower() in home or team_b.lower() in away):
                return _parse_odds(game, team_a, team_b)
    except Exception as e:
        logger.warning(f"Odds API: {e}")

    return _fallback_odds(team_a, team_b)


def _parse_odds(game: dict, team_a: str, team_b: str) -> Dict:
    bks = game.get("bookmakers", [])
    bk = next(
        (b for b in bks if "pinnacle" in b.get("key", "")),
        bks[0] if bks else None,
    )
    if not bk:
        return _fallback_odds(team_a, team_b)

    outcomes = {}
    for m in bk.get("markets", []):
        if m["key"] == "h2h":
            for o in m.get("outcomes", []):
                outcomes[o["name"]] = o["price"]

    is_a_home = team_a.lower() in game.get("home_team", "").lower()
    h_name, a_name = game["home_team"], game["away_team"]
    odds_a = outcomes.get(h_name if is_a_home else a_name, 2.5)
    odds_d = outcomes.get("Draw", 3.2)
    odds_b = outcomes.get(a_name if is_a_home else h_name, 2.8)

    raw_a, raw_d, raw_b = 1.0 / odds_a, 1.0 / odds_d, 1.0 / odds_b
    total = raw_a + raw_d + raw_b
    return {
        "odds_a": odds_a, "odds_d": odds_d, "odds_b": odds_b,
        "implied_a": round(raw_a / total, 4),
        "implied_d": round(raw_d / total, 4),
        "implied_b": round(raw_b / total, 4),
        "overround": round((total - 1.0) * 100, 2),
        "source": bk.get("title", "Bookmaker"),
    }


def _fallback_odds(team_a: str, team_b: str) -> Dict:
    tier_a = TEAM_STRENGTH_TIERS.get(team_a, 7.0)
    tier_b = TEAM_STRENGTH_TIERS.get(team_b, 7.0)
    diff = tier_a - tier_b

    if diff >= 2.5:
        return {"odds_a": 1.35, "odds_b": 6.50, "odds_d": 4.50,
                "implied_a": 0.68, "implied_b": 0.14, "implied_d": 0.18,
                "overround": 4.2, "source": "Tier-based estimate"}
    elif diff >= 1.5:
        return {"odds_a": 1.55, "odds_b": 5.00, "odds_d": 3.85,
                "implied_a": 0.60, "implied_b": 0.18, "implied_d": 0.22,
                "overround": 3.8, "source": "Tier-based estimate"}
    elif diff >= 0.5:
        return {"odds_a": 2.00, "odds_b": 3.40, "odds_d": 3.30,
                "implied_a": 0.46, "implied_b": 0.27, "implied_d": 0.27,
                "overround": 3.5, "source": "Tier-based estimate"}
    elif diff >= -0.5:
        return {"odds_a": 2.55, "odds_b": 2.55, "odds_d": 3.10,
                "implied_a": 0.37, "implied_b": 0.37, "implied_d": 0.26,
                "overround": 3.0, "source": "Tier-based estimate"}
    elif diff >= -1.5:
        return {"odds_a": 3.40, "odds_b": 2.00, "odds_d": 3.30,
                "implied_a": 0.27, "implied_b": 0.46, "implied_d": 0.27,
                "overround": 3.5, "source": "Tier-based estimate"}
    else:
        return {"odds_a": 5.00, "odds_b": 1.55, "odds_d": 3.85,
                "implied_a": 0.18, "implied_b": 0.60, "implied_d": 0.22,
                "overround": 3.8, "source": "Tier-based estimate"}


def get_live_events(team_a: str, team_b: str) -> Dict:
    if not APIFOOTBALL_KEY:
        return {"live": False, "minute": 0, "score_a": 0, "score_b": 0,
                "red_cards_a": 0, "red_cards_b": 0, "events": []}

    headers = {"x-apisports-key": APIFOOTBALL_KEY}
    try:
        r = requests.get(
            "https://v3.football.api-sports.io/fixtures",
            headers=headers, params={"live": "all"}, timeout=8,
        )
        fixtures = r.json().get("response", [])

        for f in fixtures:
            teams = f.get("teams", {})
            h = teams.get("home", {}).get("name", "").lower()
            a = teams.get("away", {}).get("name", "").lower()
            if (team_a.lower() in h or team_a.lower() in a) and \
               (team_b.lower() in h or team_b.lower() in a):
                return _parse_live(f, team_a, team_b)
    except Exception as e:
        logger.warning(f"Live events: {e}")

    return {"live": False, "minute": 0, "score_a": 0, "score_b": 0,
            "red_cards_a": 0, "red_cards_b": 0, "events": []}


def _parse_live(fixture: dict, team_a: str, team_b: str) -> Dict:
    goals = fixture.get("goals", {})
    status = fixture.get("fixture", {}).get("status", {})
    is_a_home = team_a.lower() in fixture["teams"]["home"]["name"].lower()

    score_h = goals.get("home", 0) or 0
    score_a_val = goals.get("away", 0) or 0
    score_a = score_h if is_a_home else score_a_val
    score_b = score_a_val if is_a_home else score_h

    events = []
    red_a = red_b = 0
    for ev in fixture.get("events", []):
        t = ev.get("type", "")
        team = ev.get("team", {}).get("name", "")
        is_team_a = team_a.lower() in team.lower()
        if t == "Card" and ev.get("detail") == "Red Card":
            if is_team_a:
                red_a += 1
            else:
                red_b += 1
        events.append({
            "minute": ev.get("time", {}).get("elapsed", 0),
            "type": t,
            "detail": ev.get("detail", ""),
            "team": "A" if is_team_a else "B",
            "player": ev.get("player", {}).get("name", ""),
        })

    return {
        "live": True,
        "minute": status.get("elapsed", 0) or 0,
        "score_a": score_a, "score_b": score_b,
        "red_cards_a": red_a, "red_cards_b": red_b,
        "events": events,
    }


def get_altitude(city: str, country: str) -> int:
    try:
        from geopy.geocoders import Nominatim
        geo = Nominatim(user_agent="fifa-engine")
        loc = geo.geocode(f"{city}, {country}")
        if not loc:
            return 0
        r = requests.get(
            "https://api.open-elevation.com/api/v1/lookup",
            params={"locations": f"{loc.latitude},{loc.longitude}"},
            timeout=8,
        )
        return int(r.json()["results"][0]["elevation"])
    except Exception as e:
        logger.warning(f"Altitude lookup: {e}")
        return 0


def get_news(team: str, query_type: str = "general") -> List[Dict]:
    queries = {
        "general": f"{team} football news",
        "injury": f"{team} injury training latest",
        "coach": f"{team} manager coach news",
        "morale": f"{team} training camp news",
    }
    query = queries.get(query_type, queries["general"])
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.news(query, max_results=5, timelimit="14d"))
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "date": r.get("date", ""),
                "source": r.get("source", ""),
            }
            for r in results
        ]
    except Exception as e:
        logger.warning(f"News search failed: {e}")
        return []


KNOWN_MATCHES = [
    {"home": "Mexico", "away": "South Africa", "comp": "FIFA World Cup 2026",
     "venue": "Mexico City Stadium", "city": "Mexico City", "country": "Mexico",
     "date": "2026-06-11T19:00:00Z", "status": "SCHEDULED"},
    {"home": "South Korea", "away": "Czechia", "comp": "FIFA World Cup 2026",
     "venue": "Estadio Akron", "city": "Zapopan", "country": "Mexico",
     "date": "2026-06-12T02:00:00Z", "status": "SCHEDULED"},
    {"home": "Canada", "away": "Bosnia & Herzegovina", "comp": "FIFA World Cup 2026",
     "venue": "BMO Field", "city": "Toronto", "country": "Canada",
     "date": "2026-06-12T19:00:00Z", "status": "SCHEDULED"},
    {"home": "USA", "away": "Paraguay", "comp": "FIFA World Cup 2026",
     "venue": "SoFi Stadium", "city": "Inglewood", "country": "USA",
     "date": "2026-06-13T01:00:00Z", "status": "SCHEDULED"},
    {"home": "Brazil", "away": "Morocco", "comp": "FIFA World Cup 2026",
     "venue": "MetLife Stadium", "city": "East Rutherford", "country": "USA",
     "date": "2026-06-13T19:00:00Z", "status": "SCHEDULED"},
    {"home": "Argentina", "away": "Nigeria", "comp": "FIFA World Cup 2026",
     "venue": "Hard Rock Stadium", "city": "Miami Gardens", "country": "USA",
     "date": "2026-06-14T01:00:00Z", "status": "SCHEDULED"},
    {"home": "England", "away": "Iran", "comp": "FIFA World Cup 2026",
     "venue": "AT&T Stadium", "city": "Arlington", "country": "USA",
     "date": "2026-06-14T19:00:00Z", "status": "SCHEDULED"},
    {"home": "France", "away": "Tunisia", "comp": "FIFA World Cup 2026",
     "venue": "Mercedes-Benz Stadium", "city": "Atlanta", "country": "USA",
     "date": "2026-06-15T01:00:00Z", "status": "SCHEDULED"},
]


def get_today_matches(days_ahead: int = 5) -> List[Dict]:
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")
    end_date = (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    if FOOTBALL_KEY:
        headers = {"X-Auth-Token": FOOTBALL_KEY}
        try:
            r = requests.get(
                "https://api.football-data.org/v4/matches",
                headers=headers,
                params={"dateFrom": today, "dateTo": end_date},
                timeout=8,
            )
            data = r.json()
            raw = data.get("matches", [])
        except Exception as e:
            logger.warning(f"Match fetch failed: {e}")
            raw = []
    else:
        raw = []

    if FOOTBALL_KEY and raw:
        parsed = []
        for m in raw:
            comp_name = m.get("competition", {}).get("name", "Unknown")
            home = m.get("homeTeam", {}).get("name", "Home")
            away = m.get("awayTeam", {}).get("name", "Away")
            status = m.get("status", "SCHEDULED")
            utc_date = m.get("utcDate", "")
            venue_data = m.get("venue", "") or ""
            area = m.get("area", {}).get("name", "")
            parsed.append({
                "home": home, "away": away, "comp": comp_name,
                "venue": venue_data, "city": area, "country": area,
                "date": utc_date, "status": status,
                "id": m.get("id", hash(home + away)),
            })
    else:
        parsed = []
        cutoff = now + timedelta(days=days_ahead)
        for m in KNOWN_MATCHES:
            try:
                md = datetime.fromisoformat(m["date"].replace("Z", "+00:00"))
                if now - timedelta(hours=2) <= md <= cutoff:
                    parsed.append({**m, "id": hash(m["home"] + m["away"])})
            except Exception:
                continue
        if not parsed:
            parsed = [{**m, "id": hash(m["home"] + m["away"])} for m in KNOWN_MATCHES[:3]]

    live_statuses = ("LIVE", "IN_PLAY", "PAUSED")
    parsed.sort(key=lambda m: (
        0 if m.get("status", "") in live_statuses else 1,
        m.get("date", ""),
    ))
    return parsed


def get_polymarket_price(team_a: str, team_b: str) -> Optional[Dict]:
    """
    Fetch Polymarket market price as comparison benchmark.
    Returns {outcome_a, outcome_b, draw, source} or None.
    """
    try:
        query = f"{team_a} vs {team_b}"
        encoded = requests.utils.quote(f"{team_a} {team_b} winner")
        url = f"https://clob.polymarket.com/markets?limit=5&search={encoded}"
        r = requests.get(url, timeout=8)
        if r.status_code != 200:
            return None
        data = r.json()
        markets = data if isinstance(data, list) else data.get("data", [])
        for m in markets[:3]:
            title = m.get("title", "").lower()
            if team_a.lower() in title and team_b.lower() in title:
                outcomes = m.get("outcomes", [])
                prices = m.get("prices", [])
                if outcomes and prices:
                    result = {"source": "Polymarket", "market_title": m.get("title", "")}
                    for outcome, price in zip(outcomes, prices):
                        o = str(outcome).lower()
                        p = float(price) if price else 0.5
                        if "draw" in o:
                            result["poly_draw"] = p
                        elif team_a.lower() in o:
                            result["poly_a"] = p
                        elif team_b.lower() in o:
                            result["poly_b"] = p
                    return result
        return None
    except Exception as e:
        logger.warning(f"Polymarket fetch: {e}")
        return None
