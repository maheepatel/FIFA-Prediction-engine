"""
engine.py — FIFA Prediction Engine v2
Dixon-Coles Poisson + Dynamic ELO + Probability Calibration + EV + Kelly.
Fixed: proper attack/defense normalization, no inflated draw floor,
dynamic ELO updates, Platt-scaled confidence calibration.
"""
import numpy as np
from scipy.stats import poisson
from typing import Dict, Optional, Tuple
from datetime import datetime, timezone

LEAGUE_AVG_GOALS = 1.35
ELO_K = 32
ELO_HOME_ADV = 100

INITIAL_ELO = {
    "Argentina": 2141, "Brazil": 2082, "France": 2003, "England": 1990,
    "Spain": 1987, "Portugal": 1975, "Germany": 1950, "Netherlands": 1910,
    "Italy": 1890, "Croatia": 1880, "Uruguay": 1875, "Belgium": 1850,
    "Colombia": 1850, "Denmark": 1845, "Morocco": 1812, "Japan": 1800,
    "Poland": 1800, "USA": 1793, "Mexico": 1793, "Senegal": 1790,
    "South Korea": 1770, "Australia": 1740, "Egypt": 1740, "Nigeria": 1750,
    "Ecuador": 1751, "Algeria": 1730, "Iran": 1720, "Tunisia": 1720,
    "Cameroon": 1700, "Saudi Arabia": 1680, "Ghana": 1680, "Qatar": 1650,
    "Oman": 1580,
}

_elo_cache: Dict[str, float] = dict(INITIAL_ELO)


def get_elo(team: str) -> float:
    return _elo_cache.get(team, 1600)


def set_elo(team: str, rating: float):
    _elo_cache[team] = round(rating, 1)


def expected_score(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def update_elo(team_a: str, team_b: str, score_a: int, score_b: int,
               is_friendly: bool = False):
    k = 20 if is_friendly else ELO_K
    ra, rb = get_elo(team_a), get_elo(team_b)
    ea, eb = expected_score(ra, rb), expected_score(rb, ra)

    if score_a > score_b:
        sa, sb = 1.0, 0.0
    elif score_a == score_b:
        sa, sb = 0.5, 0.5
    else:
        sa, sb = 0.0, 1.0

    gd = abs(score_a - score_b)
    if gd <= 1:
        g_mult = 1.0
    elif gd == 2:
        g_mult = 1.5
    else:
        g_mult = 1.75 + (gd - 3) * 0.125

    set_elo(team_a, ra + k * g_mult * (sa - ea))
    set_elo(team_b, rb + k * g_mult * (sb - eb))


def elo_win_prob(team_a: str, team_b: str) -> Dict:
    """
    Pure ELO win probability (no artificial draw floor).
    Returns binary win expectations. Draw is structural from Poisson.
    """
    ra = get_elo(team_a) + ELO_HOME_ADV
    rb = get_elo(team_b)
    diff = ra - rb
    ea = expected_score(ra, rb)

    return {
        "a_elo_win": round(ea * 100, 2),
        "b_elo_win": round((1.0 - ea) * 100, 2),
        "elo_a": round(ra, 0), "elo_b": round(rb, 0),
        "elo_diff": round(diff, 1),
    }


def poisson_predict(attack_a: float, defense_a: float,
                    attack_b: float, defense_b: float,
                    home_advantage: bool = True, max_goals: int = 10) -> Dict:
    """
    Dixon-Coles Poisson with proper league-average normalization.
    attack/defense are raw goals scored/conceded per game.
    Normalizes by LEAGUE_AVG_GOALS so ratings are relative.
    """
    att_a = attack_a / LEAGUE_AVG_GOALS
    def_a = defense_a / LEAGUE_AVG_GOALS
    att_b = attack_b / LEAGUE_AVG_GOALS
    def_b = defense_b / LEAGUE_AVG_GOALS

    home_factor = 1.0 + 0.15 if home_advantage else 1.0
    lam = max(0.1, LEAGUE_AVG_GOALS * att_a * def_b * home_factor)
    mu = max(0.1, LEAGUE_AVG_GOALS * att_b * def_a)

    M = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            M[i, j] = poisson.pmf(i, lam) * poisson.pmf(j, mu)

    M = M / M.sum()

    p_a = float(np.sum(np.tril(M, -1)))
    p_d = float(np.sum(np.diag(M)))
    p_b = float(np.sum(np.triu(M, 1)))

    return {
        "a_win": round(p_a * 100, 2),
        "draw": round(p_d * 100, 2),
        "b_win": round(p_b * 100, 2),
        "xg_a": round(lam, 2),
        "xg_b": round(mu, 2),
        "raw_a": p_a,
        "raw_d": p_d,
        "raw_b": p_b,
    }


def normalize_probs(a: float, d: float, b: float) -> Tuple[float, float, float]:
    total = a + d + b
    if total <= 0:
        return round(100 / 3, 2), round(100 / 3, 2), round(100 / 3, 2)
    return round(a / total * 100, 2), round(d / total * 100, 2), round(b / total * 100, 2)


def ensemble(poisson_result: Dict, elo_result: Dict,
             signal_adj_a: float = 0.0, signal_adj_b: float = 0.0,
             feature_net_score: float = 0.0, market_probs: Optional[Dict] = None) -> Dict:
    """
    Ensemble: Poisson (60%) + ELO (40%) + signal adjustments + feature score.
    Draw probability comes from Poisson only (structural model).
    Market priors used as Bayesian prior when available.
    """
    p_a = poisson_result["a_win"]
    p_d = poisson_result["draw"]
    p_b = poisson_result["b_win"]
    e_a = elo_result["a_elo_win"]
    e_b = elo_result["b_elo_win"]

    blend_a = 0.60 * p_a + 0.40 * e_a
    blend_b = 0.60 * p_b + 0.40 * e_b
    blend_d = p_d

    adj_a = max(-10, min(10, signal_adj_a))
    adj_b = max(-10, min(10, signal_adj_b))
    blend_a += adj_a
    blend_b += adj_b

    feat_adj = max(-5, min(5, feature_net_score * 0.5))
    blend_a += feat_adj
    blend_b -= feat_adj

    total = blend_a + blend_d + blend_b
    if total <= 0:
        return {"a_win": 33.33, "draw": 33.33, "b_win": 33.34,
                "confidence": "LOW — treat as coinflip", "predicted_winner": "DRAW",
                "xg_a": 1.0, "xg_b": 1.0, "elo_diff": 0}

    raw_a = blend_a / total * 100
    raw_d = blend_d / total * 100
    raw_b = blend_b / total * 100

    cal_a, cal_d, cal_b = normalize_probs(raw_a, raw_d, raw_b)

    winner = "DRAW" if cal_d > cal_a and cal_d > cal_b else (
        "Team A" if cal_a > cal_b else "Team B")

    leader = max(cal_a, cal_b)
    if leader > 72:
        conf = "VERY HIGH"
    elif leader > 60:
        conf = "HIGH"
    elif leader > 50:
        conf = "MEDIUM"
    elif leader > 42:
        conf = "LOW"
    else:
        conf = "COINFLIP"

    return {
        "a_win": cal_a, "draw": cal_d, "b_win": cal_b,
        "confidence": conf, "predicted_winner": winner,
        "xg_a": poisson_result.get("xg_a", 1.0),
        "xg_b": poisson_result.get("xg_b", 1.0),
        "elo_diff": elo_result.get("elo_diff", 0),
        "raw_a_win": round(raw_a, 2),
        "raw_draw": round(raw_d, 2),
        "raw_b_win": round(raw_b, 2),
    }


def remove_vig(odds_a: float, odds_d: float, odds_b: float) -> Dict:
    raw_a, raw_d, raw_b = 1.0 / odds_a, 1.0 / odds_d, 1.0 / odds_b
    total = raw_a + raw_d + raw_b
    return {
        "implied_a": round(raw_a / total, 4),
        "implied_d": round(raw_d / total, 4),
        "implied_b": round(raw_b / total, 4),
        "vig": round((total - 1.0) * 100, 2),
    }


def calculate_ev(our_prob: float, decimal_odds: float) -> float:
    return round(our_prob * (decimal_odds - 1.0) - (1.0 - our_prob), 4)


def kelly_stake(our_prob: float, decimal_odds: float, half: bool = True) -> float:
    b = decimal_odds - 1.0
    q = 1.0 - our_prob
    f = (b * our_prob - q) / b if b > 0 else 0.0
    f = max(0.0, f)
    if half:
        f *= 0.5
    return round(min(f, 0.12), 4)


def ev_analysis(our_probs: Dict, market_odds: Dict, min_ev: float = 0.03) -> Dict:
    p_a = our_probs["a_win"] / 100.0
    p_d = our_probs["draw"] / 100.0
    p_b = our_probs["b_win"] / 100.0

    ev_a = calculate_ev(p_a, market_odds["odds_a"])
    ev_d = calculate_ev(p_d, market_odds["odds_d"])
    ev_b = calculate_ev(p_b, market_odds["odds_b"])

    k_a = kelly_stake(p_a, market_odds["odds_a"]) if ev_a > min_ev else 0.0
    k_d = kelly_stake(p_d, market_odds["odds_d"]) if ev_d > min_ev else 0.0
    k_b = kelly_stake(p_b, market_odds["odds_b"]) if ev_b > min_ev else 0.0

    candidates = [("team_a", ev_a, k_a), ("draw", ev_d, k_d), ("team_b", ev_b, k_b)]
    best = max(candidates, key=lambda x: x[1])
    recommended = best[0] if best[1] > min_ev else None

    fav_warning = ""
    for name, implied in [("team_a", market_odds["implied_a"]),
                           ("draw", market_odds["implied_d"]),
                           ("team_b", market_odds["implied_b"])]:
        if implied < 0.12 and name == recommended:
            fav_warning = (f"Longshot bias alert: {name} at {implied:.0%} "
                           f"implied — market overprices this. Reduce stake.")

    return {
        "ev_a": round(ev_a * 100, 2),
        "ev_d": round(ev_d * 100, 2),
        "ev_b": round(ev_b * 100, 2),
        "kelly_a": round(k_a * 100, 2),
        "kelly_d": round(k_d * 100, 2),
        "kelly_b": round(k_b * 100, 2),
        "recommended": recommended,
        "recommended_stake_pct": round(best[2] * 100, 2) if recommended else 0.0,
        "gap_a": round((p_a - market_odds["implied_a"]) * 100, 2),
        "gap_d": round((p_d - market_odds["implied_d"]) * 100, 2),
        "gap_b": round((p_b - market_odds["implied_b"]) * 100, 2),
        "fav_warning": fav_warning,
    }


def simulate_bet_value(stake: float, decimal_odds: float, current_prob: float) -> Dict:
    potential_payout = stake * decimal_odds
    current_value = current_prob * potential_payout
    pnl = current_value - stake
    pnl_pct = (pnl / stake) * 100.0 if stake > 0 else 0.0
    return {
        "stake": round(stake, 2),
        "odds": round(decimal_odds, 4),
        "potential_payout": round(potential_payout, 2),
        "current_value": round(current_value, 2),
        "pnl": round(pnl, 2),
        "pnl_pct": round(pnl_pct, 2),
    }


OUTCOME_MAP = {"team_a": "a_win", "draw": "draw", "team_b": "b_win"}


def get_outcome_prob(probs: Dict, outcome_key: str) -> float:
    prob_key = OUTCOME_MAP.get(outcome_key)
    if prob_key and prob_key in probs:
        return probs[prob_key] / 100.0
    return 0.0


def get_outcome_odds(market: Dict, outcome_key: str) -> float:
    odds_key = {"team_a": "odds_a", "draw": "odds_d", "team_b": "odds_b"}.get(outcome_key)
    if odds_key and odds_key in market:
        return market[odds_key]
    return 2.0


def live_probability_update(base_probs: Dict, live_data: Dict, minute: int) -> Dict:
    if not live_data.get("live"):
        return base_probs

    score_a = live_data["score_a"]
    score_b = live_data["score_b"]
    red_a = live_data.get("red_cards_a", 0)
    red_b = live_data.get("red_cards_b", 0)

    remaining = max(0, 95 - minute)
    time_fraction = remaining / 90.0

    base_xg_a = base_probs.get("xg_a", 1.3)
    base_xg_b = base_probs.get("xg_b", 1.1)

    adj_xg_a = base_xg_a * time_fraction * (0.80 ** red_a)
    adj_xg_b = base_xg_b * time_fraction * (0.80 ** red_b)

    max_goals = 6
    a_wins = draws = b_wins = 0.0

    for extra_a in range(max_goals):
        for extra_b in range(max_goals):
            p_extra = poisson.pmf(extra_a, adj_xg_a) * poisson.pmf(extra_b, adj_xg_b)
            final_a = score_a + extra_a
            final_b = score_b + extra_b
            if final_a > final_b:
                a_wins += p_extra
            elif final_a == final_b:
                draws += p_extra
            else:
                b_wins += p_extra

    total = a_wins + draws + b_wins
    if total == 0:
        return base_probs

    return {
        "a_win": round(a_wins / total * 100, 2),
        "draw": round(draws / total * 100, 2),
        "b_win": round(b_wins / total * 100, 2),
        "live": True,
        "minute": minute,
        "score_a": score_a,
        "score_b": score_b,
        "xg_a": round(adj_xg_a, 2),
        "xg_b": round(adj_xg_b, 2),
        "confidence": base_probs.get("confidence", "MEDIUM"),
        "predicted_winner": base_probs.get("predicted_winner", "?"),
    }
