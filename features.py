"""
features.py — Feature contribution scoring system.
Each feature gets a score (-10 to +10), weight (1-10), finding, and source.
These are shown in the UI so every prediction is transparent and data-backed.
"""
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

ELO_SCALE = 400

def _safe(val, default=0):
    return val if val is not None else default

def _fmt_pct(val):
    return f"{val*100:.1f}%"

def score_to_label(score: float) -> str:
    if score > 5:    return "STRONG advantage"
    if score > 2:    return "Moderate advantage"
    if score > 0.5:  return "Slight edge"
    if score > -0.5: return "Neutral"
    if score > -2:   return "Slight disadvantage"
    if score > -5:   return "Moderate disadvantage"
    return "STRONG disadvantage"

def analyze_elo(team_a: str, team_b: str, elo_a: float, elo_b: float) -> Dict:
    diff = elo_a - elo_b
    expected = 1 / (1 + 10 ** (-diff / ELO_SCALE))
    score = (expected - 0.5) * 20
    score = max(-10, min(10, score))
    fav = team_a if diff > 0 else team_b
    return {
        "name": "ELO Rating (Team Strength)",
        "score_a": round(score, 1),
        "score_b": round(-score, 1),
        "weight": 9,
        "direction": "A" if score > 0 else "B",
        "finding": (
            f"{team_a} (ELO: {elo_a:.0f}) vs {team_b} (ELO: {elo_b:.0f}). "
            f"{'Diff: +' + str(abs(round(diff))) + ' in favour of ' + fav if diff != 0 else 'Teams are evenly matched.'} "
            f"ELO predicts {fav} as {'{:.1f}%'.format(expected*100)} favourite."
        ),
        "source": "World Football ELO ratings / FIFA ranking data"
    }

def analyze_form(team_a: str, team_b: str, form_a: Dict, form_b: Dict) -> List[Dict]:
    results = []

    # Attacking form (goals scored per game last 5)
    gs5a = _safe(form_a.get("goals_scored_5"))
    gs5b = _safe(form_b.get("goals_scored_5"))
    diff_gs = gs5a - gs5b
    score_gs = max(-8, min(8, diff_gs * 3))
    fav_gs = team_a if diff_gs > 0 else team_b if diff_gs < 0 else None
    results.append({
        "name": "Recent Attacking Form (goals scored/game last 5)",
        "score_a": round(score_gs, 1),
        "score_b": round(-score_gs, 1),
        "weight": 8,
        "direction": "A" if score_gs > 0 else "B",
        "finding": (
            f"{team_a} avg {gs5a:.2f} goals/game vs {team_b} avg {gs5b:.2f} goals/game. "
            + (f"{fav_gs} has the stronger attack recently." if fav_gs else "Both teams similar in attack.")
        ),
        "source": "Football-data.org — match results last 5 games"
    })

    # Defensive form (goals conceded per game last 5) — lower is better
    gc5a = _safe(form_a.get("goals_conceded_5"))
    gc5b = _safe(form_b.get("goals_conceded_5"))
    diff_gc = gc5b - gc5a  # reversed: lower conceded = better
    score_gc = max(-8, min(8, diff_gc * 3))
    fav_gc = team_a if gc5a < gc5b else team_b if gc5b < gc5a else None
    results.append({
        "name": "Recent Defensive Solidity (goals conceded/game last 5)",
        "score_a": round(score_gc, 1),
        "score_b": round(-score_gc, 1),
        "weight": 8,
        "direction": "A" if score_gc > 0 else "B",
        "finding": (
            f"{team_a} concede {gc5a:.2f}/game vs {team_b} concede {gc5b:.2f}/game. "
            + (f"{fav_gc} has the tighter defence." if fav_gc else "Both teams similar defensively.")
        ),
        "source": "Football-data.org — match results last 5 games"
    })

    # Form trend (wins in last 5)
    w5a = _safe(form_a.get("wins_5"))
    w5b = _safe(form_b.get("wins_5"))
    diff_w = w5a - w5b
    score_w = max(-6, min(6, diff_w * 1.5))
    fav_w = team_a if diff_w > 0 else team_b if diff_w < 0 else None
    results.append({
        "name": "Recent Winning Form (wins in last 5)",
        "score_a": round(score_w, 1),
        "score_b": round(-score_w, 1),
        "weight": 7,
        "direction": "A" if score_w > 0 else "B",
        "finding": (
            f"{team_a}: {w5a} wins, {_safe(form_a.get('draws_5'))} draws, {_safe(form_a.get('losses_5'))} losses | "
            f"{team_b}: {w5b} wins, {_safe(form_b.get('draws_5'))} draws, {_safe(form_b.get('losses_5'))} losses. "
            + (f"{fav_w} in better form." if fav_w else "Both in similar form.")
        ),
        "source": "Football-data.org — last 5 match results"
    })

    return results

def analyze_h2h(team_a: str, team_b: str, h2h: Dict) -> Dict:
    total = max(h2h.get("total", 1), 1)
    a_w = _safe(h2h.get("a_wins"))
    b_w = _safe(h2h.get("b_wins"))
    draws = _safe(h2h.get("draws"))
    a_win_rate = a_w / total
    b_win_rate = b_w / total

    h2h_score = (a_win_rate - b_win_rate) * 15
    h2h_score = max(-8, min(8, h2h_score))

    fav_h = team_a if a_win_rate > b_win_rate else team_b if b_win_rate > a_win_rate else None
    return {
        "name": "Head-to-Head Record (last " + str(total) + " meetings)",
        "score_a": round(max(-8, min(8, h2h_score)), 1),
        "score_b": round(max(-8, min(8, -h2h_score)), 1),
        "weight": 6,
        "direction": "A" if h2h_score > 0 else "B",
        "finding": (
            f"{team_a} {a_w}W — {draws}D — {b_w}W {team_b}. "
            f"Avg goals: {team_a} {h2h.get('a_goals_avg',0):.2f} — {team_b} {h2h.get('b_goals_avg',0):.2f}. "
            + (f"{fav_h} has the historical edge." if fav_h else "Evenly matched historically.")
        ),
        "source": "Football-data.org — head-to-head match history"
    }

def analyze_draw_tendency(team_a: str, team_b: str, form_a: Dict, form_b: Dict, h2h: Dict) -> Dict:
    draws_5a = _safe(form_a.get("draws_5"))
    draws_5b = _safe(form_b.get("draws_5"))
    h2h_draws = _safe(h2h.get("draws"))
    h2h_total = max(h2h.get("total", 1), 1)

    recent_draw_rate = (draws_5a + draws_5b) / 10.0
    h2h_draw_rate = h2h_draws / h2h_total

    combined_draw = (recent_draw_rate * 0.5 + h2h_draw_rate * 0.5)
    draw_score = (combined_draw - 0.2) * 12  # baseline 20% draw expectation
    draw_score = max(-6, min(6, draw_score))

    return {
        "name": "Draw Tendency Signal",
        "score_a": round(draw_score, 1),
        "score_b": round(-draw_score, 1),
        "weight": 5,
        "direction": "neutral" if abs(draw_score) < 0.5 else ("A" if draw_score < 0 else "B"),
        "finding": (
            f"Recent draws: {team_a} had {draws_5a}/5, {team_b} had {draws_5b}/5. "
            f"H2H draws: {h2h_draws}/{h2h_total} meetings. "
            + (f"Elevated draw probability ({combined_draw:.0%} rate detected) — above baseline."
               if combined_draw > 0.25 else
               f"Draw not a strong trend ({combined_draw:.0%} rate).")
        ),
        "source": "Form data + H2H history analysis"
    }

def analyze_rest(team_a: str, team_b: str, form_a: Dict, form_b: Dict) -> Dict:
    from datetime import datetime, timezone
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
    score = max(-5, min(5, diff * 0.8))
    return {
        "name": "Rest Days Advantage",
        "score_a": round(score, 1),
        "score_b": round(-score, 1),
        "weight": 6,
        "direction": "A" if score > 1 else "B" if score < -1 else "neutral",
        "finding": (
            f"{team_a}: {rest_a}d rest | {team_b}: {rest_b}d rest. "
            + (f"UEFA research: 3+ day rest gap = significant recovery advantage." if abs(diff) >= 3
               else f"Rest difference of {abs(diff)}d — minimal impact.")
        ),
        "source": "Football-data.org fixture data + UEFA workload research"
    }

def analyze_congestion(team_a: str, team_b: str, form_a: Dict, form_b: Dict) -> List[Dict]:
    results = []
    for team, side, form in [(team_a, "A", form_a), (team_b, "B", form_b)]:
        n = _safe(form.get("matches_last_21"), 2)
        if n >= 5:
            score = -4
            finding = f"{team} played {n} matches in 21 days. 30% higher injury risk per UEFA research — fatigue concern."
        elif n <= 1:
            score = 2
            finding = f"{team} played only {n} match in 21 days — well rested."
        else:
            score = 0
            finding = f"{team}: {n} matches in 21 days — normal workload."
        results.append({
            "name": "Fixture Congestion — " + team,
            "score_a": round(score if side == "A" else -score, 1),
            "score_b": round(-score if side == "A" else score, 1),
            "weight": 5,
            "direction": ("B" if side == "A" else "A") if n >= 5 else ("A" if side == "A" else "B") if n <= 1 else "neutral",
            "finding": finding,
            "source": "UEFA Player Workload Research / fixture schedule"
        })
    return results

def analyze_altitude(team_a: str, team_b: str, venue_alt: int,
                     alt_a: float, alt_b: float) -> Dict:
    gap_a = abs(venue_alt - alt_a)
    gap_b = abs(venue_alt - alt_b)
    gap_diff = gap_b - gap_a

    if gap_diff > 500:
        score = 4 if gap_diff > 1500 else 2
        return {
            "name": "Altitude Mismatch",
            "score_a": round(score, 1),
            "score_b": round(-score, 1),
            "weight": 5,
            "direction": "A",
            "finding": (
                f"{team_a} trains at {alt_a}m, venue at {venue_alt}m (gap {gap_a}m). "
                f"{team_b} gap: {gap_b}m. "
                f"FIFA Medical Research: altitude >700m mismatch affects VO2 max by 5-12%."
            ),
            "source": "Open-Elevation API + FIFA Medical Committee research"
        }
    elif gap_diff < -500:
        score = 4 if abs(gap_diff) > 1500 else 2
        return {
            "name": "Altitude Mismatch",
            "score_a": round(-score, 1),
            "score_b": round(score, 1),
            "weight": 5,
            "direction": "B",
            "finding": (
                f"{team_b} trains at {alt_b}m, venue at {venue_alt}m (gap {gap_b}m). "
                f"{team_a} gap: {gap_a}m. "
                f"FIFA Medical Research: altitude >700m mismatch affects VO2 max by 5-12%."
            ),
            "source": "Open-Elevation API + FIFA Medical Committee research"
        }
    return {
        "name": "Altitude Match",
        "score_a": 0, "score_b": 0,
        "weight": 4,
        "direction": "neutral",
        "finding": f"Venue at {venue_alt}m. Both teams' training altitudes are within {max(gap_a, gap_b)}m — minimal impact.",
        "source": "Open-Elevation API"
    }

def analyze_xg(team_a: str, team_b: str, form_a: Dict, form_b: Dict) -> Dict:
    xg_a = _safe(form_a.get("xg_scored_5"), 1.0)
    xg_b = _safe(form_b.get("xg_scored_5"), 1.0)
    xga_a = _safe(form_a.get("xg_conceded_5"), 1.0)
    xga_b = _safe(form_b.get("xg_conceded_5"), 1.0)
    xg_diff = (xg_a - xga_b) - (xg_b - xga_a)
    score = max(-6, min(6, xg_diff * 2))
    return {
        "name": "Expected Goals Differential (xG proxy)",
        "score_a": round(score, 1),
        "score_b": round(-score, 1),
        "weight": 7,
        "direction": "A" if score > 0.5 else "B" if score < -0.5 else "neutral",
        "finding": (
            f"{team_a}: {xg_a:.2f} scored, {xga_a:.2f} conceded per game. "
            f"{team_b}: {xg_b:.2f} scored, {xga_b:.2f} conceded per game. "
            f"xG differential favours {'neither team — evenly matched' if abs(xg_diff) < 0.2 else (team_a if xg_diff > 0 else team_b)}."
        ),
        "source": "Goals scored/conceded as xG proxy (actual xG requires StatsBomb data)"
    }

def analyze_poisson_prediction(team_a: str, team_b: str, poisson: Dict) -> Dict:
    xg_a = poisson.get("xg_a", 0)
    xg_b = poisson.get("xg_b", 0)
    diff = xg_a - xg_b
    score = max(-8, min(8, diff * 5))
    predicted_score = f"{team_a} {round(xg_a)}–{round(xg_b)} {team_b}"

    outcome = "Draw"
    if xg_a > xg_b + 0.3:
        outcome = f"{team_a} Win"
    elif xg_b > xg_a + 0.3:
        outcome = f"{team_b} Win"

    return {
        "name": "Poisson Predicted Scoreline",
        "score_a": round(score, 1),
        "score_b": round(-score, 1),
        "weight": 8,
        "direction": "A" if score > 0 else "B",
        "finding": (
            f"Most likely score: {predicted_score}. "
            f"xG: {team_a} {xg_a:.2f} — {team_b} {xg_b:.2f}. "
            f"Poisson model predicts {outcome}."
        ),
        "source": "Dixon-Coles Poisson model (attacking/defensive averages)"
    }

def analyze_market_sentiment(team_a: str, team_b: str, market: Dict) -> Dict:
    implied_a = _safe(market.get("implied_a"), 0.33)
    implied_b = _safe(market.get("implied_b"), 0.33)
    implied_d = _safe(market.get("implied_d"), 0.33)
    diff = implied_a - implied_b
    score = max(-6, min(6, diff * 12))
    return {
        "name": "Market Sentiment (Vig-Removed Implied Probability)",
        "score_a": round(score, 1),
        "score_b": round(-score, 1),
        "weight": 7,
        "direction": "A" if score > 0 else "B",
        "finding": (
            f"Market says: {team_a} {implied_a:.1%}, Draw {implied_d:.1%}, {team_b} {implied_b:.1%}. "
            f"Overround: {_safe(market.get('overround')):.1f}%. "
            f"Market prices {team_a if implied_a > implied_b else team_b} as favourite."
        ),
        "source": f"{market.get('source', 'Bookmaker odds')} — the-odds-api.com"
    }

def run_feature_analysis(team_a: str, team_b: str, elo_a: float, elo_b: float,
                          form_a: Dict, form_b: Dict, h2h: Dict, poisson: Dict,
                          market: Dict, venue_alt: int, alt_a: float, alt_b: float) -> Dict:
    all_features = []

    def add(f):
        if isinstance(f, list):
            all_features.extend(f)
        elif f:
            all_features.append(f)

    add(analyze_elo(team_a, team_b, elo_a, elo_b))
    add(analyze_form(team_a, team_b, form_a, form_b))
    add(analyze_h2h(team_a, team_b, h2h))
    add(analyze_draw_tendency(team_a, team_b, form_a, form_b, h2h))
    add(analyze_rest(team_a, team_b, form_a, form_b))
    add(analyze_congestion(team_a, team_b, form_a, form_b))
    add(analyze_altitude(team_a, team_b, venue_alt, alt_a, alt_b))
    add(analyze_xg(team_a, team_b, form_a, form_b))
    add(analyze_poisson_prediction(team_a, team_b, poisson))
    add(analyze_market_sentiment(team_a, team_b, market))

    total_weight = sum(f.get("weight", 5) for f in all_features)
    weighted_score_a = sum(f.get("score_a", 0) * f.get("weight", 5) for f in all_features)
    weighted_score_b = sum(f.get("score_b", 0) * f.get("weight", 5) for f in all_features)
    net_score = (weighted_score_a - weighted_score_b) / total_weight if total_weight > 0 else 0
    net_score = max(-10, min(10, net_score))

    return {
        "features": all_features,
        "net_score": round(net_score, 2),
        "total_weight": total_weight,
        "team_a_total": round(weighted_score_a / total_weight, 2) if total_weight > 0 else 0,
        "team_b_total": round(weighted_score_b / total_weight, 2) if total_weight > 0 else 0,
    }
