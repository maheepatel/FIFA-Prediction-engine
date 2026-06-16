"""
app.py — FIFA Prediction Engine v2 UI
Modern card-based dashboard with model comparison, live updates.
streamlit run app.py
"""
import streamlit as st
import plotly.graph_objects as go
import time
from datetime import datetime, timezone

from data import (
    get_team_form, get_h2h, get_odds, get_live_events,
    get_today_matches, get_polymarket_price,
)
from engine import (
    elo_win_prob, poisson_predict, ensemble, ev_analysis,
    live_probability_update, simulate_bet_value,
    get_outcome_prob, get_outcome_odds, get_elo,
)
from signals import run_all_signals
from features import run_feature_analysis, score_to_label
from report import build_report

st.set_page_config(
    page_title="FIFA Prediction Engine",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

NEWSPRINT_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Playfair+Display:ital,wght@0,400;0,600;0,700;0,900;1,400&family=Lora:ital,wght@0,400;0,600;1,400&display=block');

    * { border-radius: 0 !important; }

    .stApp {
        background-color: #F9F9F7;
        color: #111111;
        font-family: 'Inter', 'Helvetica Neue', sans-serif;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='4' height='4' viewBox='0 0 4 4'%3E%3Cpath fill='%23111111' fill-opacity='0.04' d='M1 3h1v1H1V3zm2-2h1v1H3V1z'%3E%3C/path%3E%3C/svg%3E");
    }
    .main .block-container { max-width: 1280px; padding: 0 1rem; }
    h1, h2, h3 { font-family: 'Playfair Display', 'Times New Roman', serif !important; color: #111111 !important; }
    h1 { font-size: 3rem; font-weight: 900; letter-spacing: -0.02em; line-height: 0.9; }
    h2 { font-size: 2.25rem; font-weight: 700; letter-spacing: -0.01em; }
    h3 { font-size: 1.5rem; font-weight: 700; }

    .st-bx { background-color: #F9F9F7; border: 1px solid #111111; padding: 1.5rem; margin-bottom: 1rem; }
    div[data-testid="metric-container"] { background: #F9F9F7; border: 1px solid #111111; padding: 1rem; margin: 0; }
    div[data-testid="metric-container"] label { color: #737373 !important; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.15em; font-family: 'JetBrains Mono', monospace; }
    div[data-testid="metric-container"] div[data-testid="metric-value"] { color: #111111 !important; font-size: 1.8rem !important; font-weight: 700; font-family: 'Playfair Display', serif; }
    div[data-testid="stSelectbox"] { border: none !important; background: transparent !important; }
    div[data-testid="stSelectbox"] > div { background: #F9F9F7; border: 1px solid #111111; min-height: 44px; display: flex; align-items: center; padding: 0 0.5rem; }
    div[data-testid="stSelectbox"] div[data-baseweb="select"], 
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div, 
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div > div,
    div[data-testid="stSelectbox"] div[role="button"] {
        border: none !important; background: transparent !important; box-shadow: none !important; outline: none !important; 
    }
    div[data-testid="stSelectbox"] [data-baseweb="select"] span { background: transparent !important; }
    div[data-testid="stSelectbox"] [data-baseweb="popover"] { background: #F9F9F7 !important; border: 1px solid #111111 !important; box-shadow: 4px 4px 0px 0px #111111 !important; margin-top: 2px !important; }
    div[data-testid="stSelectbox"] [data-baseweb="popover"] li { font-family: 'Inter', sans-serif !important; font-size: 0.8rem !important; border-bottom: 1px solid #E5E5E0 !important; padding: 0.5rem 0.75rem !important; }
    div[data-testid="stSelectbox"] [data-baseweb="popover"] li:hover { background: #F0F0F0 !important; }
    div[data-testid="stSelectbox"] svg { fill: #111111 !important; }
    div[data-testid="stSelectbox"] label { font-family: 'Inter', sans-serif !important; font-size: 0.65rem !important; text-transform: uppercase !important; letter-spacing: 0.1em !important; color: #737373 !important; }
    .st-bw { background-color: #F9F9F7; }
    .stButton > button {
        font-family: 'Inter', sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid #111111 !important;
        background: #111111 !important;
        color: #F9F9F7 !important;
        transition: all 0.2s ease;
        min-height: 44px;
    }
    .stButton > button:hover {
        background: #F9F9F7 !important;
        color: #111111 !important;
    }
    .stButton > button[kind="secondary"] {
        background: #F9F9F7 !important;
        color: #111111 !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background: #111111 !important;
        color: #F9F9F7 !important;
    }
    .st-emotion-cache-1y4p8pa { padding: 2rem 1rem; }
    .stProgress > div > div > div { background: #111111 !important; }
    footer { display: none; }
    #MainMenu { display: none; }
    header { display: none; }
    div[data-testid="stExpander"] { background: #F9F9F7; border: 1px solid #111111; }
    .stAlert { background-color: #F9F9F7; border: 1px solid #111111; color: #111111; font-family: 'Inter', sans-serif; font-size: 0.8rem; }
    .stAlert p { font-family: 'Inter', sans-serif; font-size: 0.8rem; }
    hr { border-color: #111111; margin: 2rem 0; border-width: 2px 0 0 0; }

    .conf-very-high { color: #111111; font-weight: 700; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.15em; }
    .conf-high { color: #404040; font-weight: 600; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.15em; }
    .conf-medium { color: #737373; font-weight: 600; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.15em; }
    .conf-low { color: #525252; font-weight: 600; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.15em; }
    .conf-coinflip { color: #CC0000; font-weight: 700; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.15em; }

    .prob-card { border: 2px solid #111111; padding: 1.5rem; text-align: center; background: #F9F9F7; }
    .prob-card-title { font-family: 'Inter', sans-serif; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.2em; color: #737373; margin-bottom: 0.25rem; }
    .prob-value { font-family: 'Playfair Display', serif; font-size: 3rem; font-weight: 900; line-height: 1; }
    .prob-bar { height: 4px; background: #E5E5E0; margin-top: 0.75rem; }
    .prob-bar-fill { height: 100%; background: #111111; }
    .prob-meta { font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; color: #737373; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.4rem; display: flex; justify-content: space-between; }

    .hard-shadow-hover { transition: all 0.2s ease; }
    .hard-shadow-hover:hover { box-shadow: 4px 4px 0px 0px #111111; transform: translate(-2px, -2px); }

    .edition { font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; text-transform: uppercase; letter-spacing: 0.25em; color: #737373; border-bottom: 2px solid #111111; padding-bottom: 0.5rem; margin-bottom: 1rem; }

    .badge { display: inline-block; padding: 0.15rem 0.5rem; font-size: 0.6rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.15em; font-family: 'JetBrains Mono', monospace; border: 1px solid #111111; }
    .badge-red { background: #CC0000; color: #F9F9F7; border-color: #CC0000; }
    .badge-black { background: #111111; color: #F9F9F7; }

    .section-label { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.25em; color: #737373; margin-bottom: 0.25rem; }

    .signal-card { border: 1px solid #111111; padding: 0.75rem 1rem; margin-bottom: 0.5rem; background: #F9F9F7; }

    .ornament { font-family: 'Playfair Display', serif; font-size: 1.5rem; color: #A3A3A3; letter-spacing: 1em; text-align: center; padding: 0.75rem 0; }

    .stat-card { border: 1px solid #111111; padding: 1rem; background: #F9F9F7; }
    .stat-label { font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; text-transform: uppercase; letter-spacing: 0.15em; color: #737373; margin: 0; }
    .stat-value { font-family: 'Playfair Display', serif; font-size: 1.1rem; font-weight: 700; margin: 0.15rem 0; color: #111111; }
    .stat-sub { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: #737373; margin: 0; }

    .ev-card { border: 2px solid #111111; padding: 1rem; text-align: center; background: #F9F9F7; }
    .ev-value { font-family: 'Playfair Display', serif; font-size: 1.8rem; font-weight: 900; }
    .ev-label { font-family: 'Inter', sans-serif; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.15em; color: #737373; margin-top: 0.25rem; }
    .ev-sub { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #737373; margin-top: 0.15rem; }

    .ticker { background: #111111; color: #F9F9F7; padding: 0.5rem 1rem; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; }

    input, textarea, .stNumberInput input, .stNumberInput div[data-baseweb="input"] { border-radius: 0 !important; border-bottom: 2px solid #111111 !important; background: transparent !important; font-family: 'JetBrains Mono', monospace !important; }
    .stNumberInput div[data-baseweb="input"] { border: none !important; box-shadow: none !important; }
    .stNumberInput button { border: 1px solid #111111 !important; background: #F9F9F7 !important; border-radius: 0 !important; color: #111111 !important; min-height: 44px; min-width: 44px; }
    .stNumberInput button:hover { background: #111111 !important; color: #F9F9F7 !important; }
    input:focus, textarea:focus { background: #F0F0F0 !important; outline: none !important; }

    @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }
    .live-dot { width: 10px; height: 10px; background: #CC0000; display: inline-block; animation: pulse 1.5s infinite; }
</style>
"""
st.markdown(NEWSPRINT_CSS, unsafe_allow_html=True)

if "matches" not in st.session_state:
    st.session_state["matches"] = get_today_matches()
if "match_idx" not in st.session_state:
    st.session_state["match_idx"] = 0
if "prob_history" not in st.session_state:
    st.session_state["prob_history"] = []
if "prediction_records" not in st.session_state:
    st.session_state["prediction_records"] = {}


def get_match():
    ms = st.session_state["matches"]
    idx = st.session_state["match_idx"]
    if 0 <= idx < len(ms):
        return ms[idx]
    return None


# ── SIDEBAR ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        "<div class='edition'>Vol. 1 | June 2026 | World Cup Edition</div>"
        "<h1 style='font-size:1.75rem; margin-bottom:0; font-weight:900;'>FIFA<br>Predict</h1>"
        "<p style='color:#737373; font-size:0.75rem; margin-top:0.25rem; font-family:\"JetBrains Mono\",monospace; text-transform:uppercase; letter-spacing:0.1em;'>"
        "Dixon-Coles Poisson + Dynamic ELO + Ensemble</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='margin:1rem 0;'>", unsafe_allow_html=True)

    match = get_match()
    matches = st.session_state["matches"]
    match_labels = []
    for m in matches:
        sts = m.get("status", "")
        if sts == "FINISHED":
            sh = m.get("score_home", "")
            sa = m.get("score_away", "")
            lbl = f"{m['home']} {sh}–{sa} {m['away']} (FT)"
        elif sts in ("LIVE", "IN_PLAY", "PAUSED"):
            lbl = f"🔴 {m['home']} vs {m['away']}"
        else:
            lbl = f"{m['home']} vs {m['away']}"
        match_labels.append(lbl)

    team_a = "Team A"
    team_b = "Team B"
    competition = "FIFA World Cup 2026"
    venue_city = "Unknown"
    venue_country = "Unknown"
    is_live = False
    is_finished = False

    if matches:
        sel = st.selectbox(
            "Select Match",
            range(len(match_labels)),
            format_func=lambda i: match_labels[i],
            index=st.session_state["match_idx"],
        )
        if sel != st.session_state["match_idx"]:
            st.session_state["match_idx"] = sel
            st.session_state["prob_history"] = []
            st.rerun()

        match = get_match()
        if match:
            team_a = match["home"]
            team_b = match["away"]
            competition = match.get("comp", "FIFA World Cup 2026")
            venue_city = match.get("city", "Unknown")
            venue_country = match.get("country", "Unknown")
            match_status = match.get("status", "SCHEDULED")
            is_live = match_status in ("LIVE", "IN_PLAY", "PAUSED")
            is_finished = match_status == "FINISHED"
        else:
            first = matches[0]
            team_a = first["home"]
            team_b = first["away"]
            competition = first.get("comp", competition)
            venue_city = first.get("city", venue_city)
            venue_country = first.get("country", venue_country)
            is_live = first.get("status", "SCHEDULED") in ("LIVE", "IN_PLAY", "PAUSED")
            is_finished = first.get("status", "SCHEDULED") == "FINISHED"
            st.session_state["match_idx"] = 0
    else:
        st.warning("No matches available.")

    st.session_state["team_a"] = team_a
    st.session_state["team_b"] = team_b

    bankroll = st.number_input("Bankroll ($)", value=1000, step=100)

    st.markdown("<hr style='margin:1rem 0;'>", unsafe_allow_html=True)
    st.markdown("<p class='section-label'>Betting Simulator</p>", unsafe_allow_html=True)
    if "active_bet" in st.session_state:
        ab = st.session_state["active_bet"]
        st.info(f"Active: {ab['outcome_label']} @ ${ab['stake']:.0f}")
        if st.button("Cash Out / Clear", use_container_width=True):
            del st.session_state["active_bet"]
            st.session_state["prob_history"] = []
            st.rerun()
    else:
        bet_outcome = st.selectbox("Bet on", ["Team A", "Draw", "Team B"])
        bet_outcome_key = {"Team A": "team_a", "Draw": "draw", "Team B": "team_b"}[
            bet_outcome
        ]
        bet_label = {"Team A": team_a, "Draw": "Draw", "Team B": team_b}[bet_outcome]
        bet_stake = st.number_input("Virtual Stake ($)", value=100, min_value=10, step=10)
        if st.button("Place Bet", use_container_width=True, type="secondary"):
            if "market" in st.session_state:
                odds = get_outcome_odds(st.session_state["market"], bet_outcome_key)
                st.session_state["active_bet"] = {
                    "outcome": bet_outcome_key,
                    "outcome_label": bet_label,
                    "stake": bet_stake,
                    "odds": odds,
                    "potential_payout": round(bet_stake * odds, 2),
                    "placed_at": datetime.now(),
                }
                st.rerun()
            else:
                st.error("Waiting for prediction data...")

    st.markdown("<hr style='margin:1rem 0;'>", unsafe_allow_html=True)
    st.markdown("<p style='font-family:\"JetBrains Mono\",monospace; font-size:0.6rem; text-transform:uppercase; letter-spacing:0.15em; color:#737373;'>Auto-refresh every 30s</p>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Refresh", use_container_width=True):
            st.session_state["matches"] = get_today_matches()
            st.rerun()
    with col2:
        if st.button("Reset ELO", use_container_width=True):
            from engine import _elo_cache, INITIAL_ELO
            _elo_cache.clear()
            _elo_cache.update(INITIAL_ELO)
            st.rerun()


# ── MAIN ─────────────────────────────────────────────────────────────────────

if not matches:
    st.warning("No matches found. Add API keys to .env or check your connection.")
    st.stop()


with st.spinner("Loading prediction model..."):
    form_a = get_team_form(team_a)
    form_b = get_team_form(team_b)
    h2h = get_h2h(team_a, team_b)
    market = get_odds(team_a, team_b, competition)
    sigs = run_all_signals(team_a, team_b, venue_city, venue_country, form_a, form_b)

    live_data = {}
    if is_live:
        live_data = get_live_events(team_a, team_b)
        if live_data.get("live"):
            match_status = "LIVE"

    elo = elo_win_prob(team_a, team_b)
    poisson = poisson_predict(
        attack_a=form_a["goals_scored_5"],
        defense_a=form_a["goals_conceded_5"],
        attack_b=form_b["goals_scored_5"],
        defense_b=form_b["goals_conceded_5"],
    )
    feat_analysis = run_feature_analysis(
        team_a, team_b,
        elo.get("elo_a", 1650), elo.get("elo_b", 1650),
        form_a, form_b, h2h, poisson, market,
        sigs.get("venue_alt", 0),
        sigs.get("alt_a", 100), sigs.get("alt_b", 100),
    )
    probs = ensemble(
        poisson, elo,
        sigs["adj_a"], sigs["adj_b"],
        feature_net_score=feat_analysis["net_score"],
    )

    poly_prices = get_polymarket_price(team_a, team_b)

    if is_live and live_data.get("live"):
        probs = live_probability_update(probs, live_data, live_data.get("minute", 0))

    ev = ev_analysis(probs, market)

    st.session_state["probs"] = probs
    st.session_state["market"] = market
    st.session_state["is_live"] = is_live
    st.session_state["live_data"] = live_data

    conf = probs.get("confidence", "MEDIUM")

    match_id = match.get("id", hash(team_a + team_b)) if match else hash(team_a + team_b)
    match_key = f"{match_id}"
    if match_key not in st.session_state["prediction_records"]:
        st.session_state["prediction_records"][match_key] = {
            "home": team_a, "away": team_b,
            "competition": competition,
            "venue": f"{venue_city}, {venue_country}",
            "date": match.get("date", "") if match else "",
            "model_a": probs["a_win"],
            "model_d": probs["draw"],
            "model_b": probs["b_win"],
            "confidence": conf,
            "predicted_winner": probs.get("predicted_winner", "?"),
            "model_xg_a": probs.get("xg_a", "-"),
            "model_xg_b": probs.get("xg_b", "-"),
            "market_a": market.get("implied_a", 0) * 100,
            "market_d": market.get("implied_d", 0) * 100,
            "market_b": market.get("implied_b", 0) * 100,
            "actual_score_h": None,
            "actual_score_a": None,
            "actual_winner": None,
            "status": match_status,
        }
    rec = st.session_state["prediction_records"][match_key]
    if is_finished:
        rec["status"] = "FINISHED"
        act_h = match.get("score_home") if match else None
        act_a = match.get("score_away") if match else None
        if act_h is not None and act_a is not None:
            rec["actual_score_h"] = act_h
            rec["actual_score_a"] = act_a
            if act_h > act_a:
                rec["actual_winner"] = "home"
            elif act_h == act_a:
                rec["actual_winner"] = "draw"
            else:
                rec["actual_winner"] = "away"
    elif is_live:
        rec["status"] = "LIVE"
        ld = live_data or {}
        if ld.get("live"):
            rec["actual_score_h"] = ld.get("score_a", 0)
            rec["actual_score_a"] = ld.get("score_b", 0)

    now_min = live_data.get("minute", 0) if is_live and live_data.get("live") else 0
    entry = {
        "minute": now_min,
        "a": probs["a_win"],
        "d": probs["draw"],
        "b": probs["b_win"],
    }
    if not st.session_state["prob_history"] or st.session_state["prob_history"][-1] != entry:
        st.session_state["prob_history"].append(entry)
    st.session_state["prob_history"] = st.session_state["prob_history"][-100:]


# ── MATCH HEADER ─────────────────────────────────────────────────────────────

venue_str = match.get("venue", f"{venue_city}, {venue_country}") if match else "—"
match_date = match.get("date", "") if match else ""

col_hdr, col_live = st.columns([3, 1])
with col_hdr:
    if is_finished:
        act_h = match.get("score_home") if match else None
        act_a = match.get("score_away") if match else None
        scored = f"{act_h}–{act_a}" if act_h is not None and act_a is not None else "—"
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:0.75rem;flex-wrap:wrap;'>"
            f"<h2 style='margin:0; font-size:2.5rem;'>{team_a} vs {team_b}</h2>"
            f"<span class='badge badge-black'>FINAL</span>"
            f"<span style='font-family:\"Playfair Display\",serif; font-size:2rem; font-weight:900;'>{scored}</span>"
            f"</div>"
            f"<p style='color:#737373;margin:0.25rem 0; font-family:\"JetBrains Mono\",monospace; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em;'>{venue_str} | {competition}</p>",
            unsafe_allow_html=True,
        )
    elif is_live and live_data.get("live"):
        ld = live_data
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:0.75rem;flex-wrap:wrap;'>"
            f"<span class='live-dot'></span>"
            f"<h2 style='margin:0; font-size:2.5rem;'>{team_a} vs {team_b}</h2>"
            f"<span class='badge badge-red'>LIVE {ld['minute']}'</span>"
            f"<span style='font-family:\"Playfair Display\",serif; font-size:2rem; font-weight:700;'>{ld['score_a']}–{ld['score_b']}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<h2 style='margin:0; font-size:2.5rem;'>{team_a} vs {team_b}</h2>"
            f"<p style='color:#737373;margin:0.25rem 0; font-family:\"JetBrains Mono\",monospace; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em;'>{venue_str} | {competition}</p>",
            unsafe_allow_html=True,
        )

with col_live:
    if is_live and live_data.get("live"):
        pass
    else:
        try:
            md = datetime.fromisoformat(match_date.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            delta = md - now
            if delta.total_seconds() > 0:
                hrs = int(delta.total_seconds() // 3600)
                mins = int((delta.total_seconds() % 3600) // 60)
                st.markdown(
                    f"<div style='text-align:right; border:2px solid #111111; padding:0.75rem 1rem;'>"
                    f"<span style='font-family:\"JetBrains Mono\",monospace; font-size:0.6rem; text-transform:uppercase; letter-spacing:0.15em; color:#737373;'>Kick-off in</span><br>"
                    f"<span style='font-family:\"Playfair Display\",serif; font-size:1.8rem; font-weight:900;'>{hrs}h {mins}m</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        except Exception:
            pass


# ── SETTLED MATCH ───────────────────────────────────────────────────────────

if is_finished:
    match_id = match.get("id", hash(team_a + team_b)) if match else hash(team_a + team_b)
    match_key = f"{match_id}"
    rec = st.session_state["prediction_records"].get(match_key, {})
    if rec.get("actual_score_h") is not None:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<p class='section-label'>Settled</p>", unsafe_allow_html=True)
        st.markdown(
            f"<h3>Prediction vs Result</h3>"
            f"<span style='color:#737373; font-family:\"JetBrains Mono\",monospace; font-size:0.65rem; text-transform:uppercase; letter-spacing:0.1em;'>"
            f"Pre-match prediction vs actual outcome</span>",
            unsafe_allow_html=True,
        )

        model_pred = rec.get("predicted_winner", "?")
        model_pct = max(rec.get("model_a", 0), rec.get("model_d", 0), rec.get("model_b", 0))
        act_h = rec.get("actual_score_h", 0)
        act_a = rec.get("actual_score_a", 0)
        if act_h > act_a:
            actual_winner_key = "home"
            actual_label = team_a
        elif act_h == act_a:
            actual_winner_key = "draw"
            actual_label = "Draw"
        else:
            actual_winner_key = "away"
            actual_label = team_b

        model_correct = False
        if actual_winner_key == "home" and model_pred in ("Team A", team_a):
            model_correct = True
        elif actual_winner_key == "draw" and model_pred == "DRAW":
            model_correct = True
        elif actual_winner_key == "away" and model_pred in ("Team B", team_b):
            model_correct = True

        col_sa, col_sb, col_sc = st.columns(3)
        with col_sa:
            st.markdown(
                f"<div class='prob-card'>"
                f"<div class='prob-card-title'>Actual Result</div>"
                f"<div style='font-family:\"Playfair Display\",serif; font-size:2.5rem; font-weight:900;'>{act_h} – {act_a}</div>"
                f"<div class='prob-meta'>{actual_label}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with col_sb:
            st.markdown(
                f"<div class='prob-card'>"
                f"<div class='prob-card-title'>Model Predicted</div>"
                f"<div style='font-family:\"Playfair Display\",serif; font-size:2.5rem; font-weight:900;'>{model_pct:.1f}%</div>"
                f"<div class='prob-meta'>{model_pred}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with col_sc:
            verdict = "CORRECT" if model_correct else "INCORRECT"
            v_color = "#111111" if model_correct else "#CC0000"
            st.markdown(
                f"<div class='ev-card' style='border-color:{v_color};'>"
                f"<div class='ev-label'>Verdict</div>"
                f"<div style='font-family:\"Playfair Display\",serif; font-size:1.5rem; font-weight:900; color:{v_color};'>{verdict}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown(f"<p class='section-label' style='margin-top:0.5rem;'>Model Breakdown</p>", unsafe_allow_html=True)
            st.markdown(
                f"<div style='display:grid; grid-template-columns:1fr 1fr 1fr; border:1px solid #111111; text-align:center;'>"
                f"<div style='padding:0.5rem; border-right:1px solid #111111; font-family:\"JetBrains Mono\",monospace; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.1em; color:#737373;'>Outcome</div>"
                f"<div style='padding:0.5rem; border-right:1px solid #111111; font-family:\"JetBrains Mono\",monospace; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.1em; color:#737373;'>Model</div>"
                f"<div style='padding:0.5rem; font-family:\"JetBrains Mono\",monospace; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.1em; color:#737373;'>Market</div>"
                f"<div style='padding:0.4rem; border-right:1px solid #111111; border-top:1px solid #111111; font-family:\"Playfair Display\",serif; font-weight:700;'>{team_a}</div>"
                f"<div style='padding:0.4rem; border-right:1px solid #111111; border-top:1px solid #111111;'>{rec.get('model_a',0):.1f}%</div>"
                f"<div style='padding:0.4rem; border-top:1px solid #111111;'>{rec.get('market_a',0):.1f}%</div>"
                f"<div style='padding:0.4rem; border-right:1px solid #111111; border-top:1px solid #111111; font-family:\"Playfair Display\",serif; font-weight:700;'>Draw</div>"
                f"<div style='padding:0.4rem; border-right:1px solid #111111; border-top:1px solid #111111;'>{rec.get('model_d',0):.1f}%</div>"
                f"<div style='padding:0.4rem; border-top:1px solid #111111;'>{rec.get('market_d',0):.1f}%</div>"
                f"<div style='padding:0.4rem; border-right:1px solid #111111; border-top:1px solid #111111; font-family:\"Playfair Display\",serif; font-weight:700;'>{team_b}</div>"
                f"<div style='padding:0.4rem; border-right:1px solid #111111; border-top:1px solid #111111;'>{rec.get('model_b',0):.1f}%</div>"
                f"<div style='padding:0.4rem; border-top:1px solid #111111;'>{rec.get('market_b',0):.1f}%</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with col_s2:
            st.markdown(f"<p class='section-label' style='margin-top:0.5rem;'>Match Info</p>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='stat-card'>"
                f"<p class='stat-label'>Confidence</p>"
                f"<p class='stat-value'>{rec.get('confidence', '—')}</p>"
                f"<p class='stat-label' style='margin-top:0.5rem;'>Predicted Score</p>"
                f"<p class='stat-value'>{rec.get('model_xg_a','-')}–{rec.get('model_xg_b','-')}</p>"
                f"<p class='stat-label' style='margin-top:0.5rem;'>Actual Score</p>"
                f"<p class='stat-value'>{act_h}–{act_a}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ── PROBABILITY CARDS ────────────────────────────────────────────────────────

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p class='section-label'>Prediction</p>", unsafe_allow_html=True)
st.markdown("<h3>Win Probability</h3>", unsafe_allow_html=True)

conf_class = {
    "VERY HIGH": "conf-very-high",
    "HIGH": "conf-high",
    "MEDIUM": "conf-medium",
    "LOW": "conf-low",
    "COINFLIP": "conf-coinflip",
}.get(conf, "conf-medium")

col_a, col_d, col_b = st.columns(3)
with col_a:
    pct_a = probs["a_win"]
    st.markdown(
        f"<div class='prob-card'>"
        f"<div class='prob-card-title'>{team_a}</div>"
        f"<div class='prob-value'>{pct_a:.1f}%</div>"
        f"<div class='prob-bar'><div class='prob-bar-fill' style='width:{min(pct_a,100)}%;'></div></div>"
        f"<div class='prob-meta'><span>xG: {probs.get('xg_a','-')}</span><span>ELO: {elo.get('elo_a',0):.0f}</span></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

with col_d:
    pct_d = probs["draw"]
    st.markdown(
        f"<div class='prob-card'>"
        f"<div class='prob-card-title'>Draw</div>"
        f"<div class='prob-value'>{pct_d:.1f}%</div>"
        f"<div class='prob-bar'><div class='prob-bar-fill' style='width:{min(pct_d,100)}%;'></div></div>"
        f"<div class='prob-meta'><span>{probs.get('predicted_winner','?')}</span></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

with col_b:
    pct_b = probs["b_win"]
    st.markdown(
        f"<div class='prob-card'>"
        f"<div class='prob-card-title'>{team_b}</div>"
        f"<div class='prob-value'>{pct_b:.1f}%</div>"
        f"<div class='prob-bar'><div class='prob-bar-fill' style='width:{min(pct_b,100)}%;'></div></div>"
        f"<div class='prob-meta'><span>xG: {probs.get('xg_b','-')}</span><span>ELO: {elo.get('elo_b',0):.0f}</span></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown(
    f"<div style='text-align:center; padding:0.75rem; border:1px solid #111111; margin-top:0.5rem;'>"
    f"<span class='{conf_class}'>{'●' if conf in ('VERY HIGH','HIGH') else '◐' if conf == 'MEDIUM' else '○' if conf == 'LOW' else '◉'} Confidence: {conf}</span>"
    f"<span style='color:#737373; margin-left:1.5rem; font-family:\"JetBrains Mono\",monospace; font-size:0.7rem;'>Predicted: {probs.get('xg_a','-')}–{probs.get('xg_b','-')}</span>"
    f"<span style='color:#737373; margin-left:1.5rem; font-family:\"JetBrains Mono\",monospace; font-size:0.7rem;'>ELO gap: {probs.get('elo_diff',0):+.0f}</span>"
    f"</div>",
    unsafe_allow_html=True,
)

st.markdown("<div class='ornament'>✦ ✦ ✦</div>", unsafe_allow_html=True)


# ── MODEL vs MARKET ──────────────────────────────────────────────────────────

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p class='section-label'>Benchmarking</p>", unsafe_allow_html=True)
st.markdown("<h3>Model vs Market</h3>", unsafe_allow_html=True)

col_chart, col_meta = st.columns([3, 1])
with col_chart:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Our Model",
        x=[team_a, "Draw", team_b],
        y=[probs["a_win"], probs["draw"], probs["b_win"]],
        marker_color=["#111111", "#737373", "#111111"],
        opacity=0.85,
        marker_pattern_shape="/",
    ))
    fig.add_trace(go.Bar(
        name=f"Market ({market.get('source', 'Odds')})",
        x=[team_a, "Draw", team_b],
        y=[
            market["implied_a"] * 100,
            market["implied_d"] * 100,
            market["implied_b"] * 100,
        ],
        marker_color=["#E5E5E0", "#E5E5E0", "#E5E5E0"],
        opacity=1,
    ))
    if poly_prices and "poly_a" in poly_prices:
        fig.add_trace(go.Scatter(
            name="Polymarket",
            x=[team_a, "Draw", team_b],
            y=[
                poly_prices.get("poly_a", 0) * 100,
                poly_prices.get("poly_draw", 0) * 100,
                poly_prices.get("poly_b", 0) * 100,
            ],
            mode="markers",
            marker=dict(color="#CC0000", size=12, symbol="diamond"),
        ))
    fig.update_layout(
        barmode="group",
        yaxis_title="Probability (%)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=320,
        font=dict(color="#111111", family="Inter, sans-serif"),
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", y=1.1, font=dict(size=11)),
        hovermode="x unified",
        bargap=0.15,
    )
    fig.update_xaxes(gridcolor="#E5E5E0", tickfont=dict(family="Playfair Display, serif", size=13))
    fig.update_yaxes(gridcolor="#E5E5E0", range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

with col_meta:
    if poly_prices:
        st.markdown(
            f"<div style='border:1px solid #111111;padding:1rem;'>"
            f"<p style='font-family:\"JetBrains Mono\",monospace; font-size:0.6rem; text-transform:uppercase; letter-spacing:0.15em; color:#CC0000; font-weight:600; margin:0;'>Polymarket</p>"
            f"<p style='font-size:0.7rem;color:#737373; font-family:\"Inter\",sans-serif;'>{poly_prices.get('market_title', '')[:40]}</p>"
            f"<p style='font-family:\"JetBrains Mono\",monospace; font-size:0.8rem;'>{team_a}: <strong>{poly_prices.get('poly_a',0)*100:.1f}%</strong></p>"
            f"<p style='font-family:\"JetBrains Mono\",monospace; font-size:0.8rem;'>Draw: <strong>{poly_prices.get('poly_draw',0)*100:.1f}%</strong></p>"
            f"<p style='font-family:\"JetBrains Mono\",monospace; font-size:0.8rem;'>{team_b}: <strong>{poly_prices.get('poly_b',0)*100:.1f}%</strong></p>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='border:1px solid #E5E5E0;padding:1rem;text-align:center;'>"
            f"<p style='color:#737373; font-family:\"JetBrains Mono\",monospace; font-size:0.65rem; text-transform:uppercase; letter-spacing:0.1em;'>Polymarket</p>"
            f"<p style='color:#A3A3A3; font-family:\"Inter\",sans-serif; font-size:0.75rem;'>Not available</p>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ── EV + KELLY ───────────────────────────────────────────────────────────────

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p class='section-label'>Edge Analysis</p>", unsafe_allow_html=True)
st.markdown("<h3>Expected Value &amp; Kelly Stake</h3>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
for col, label, ev_val, kelly_val in [
    (c1, team_a, ev["ev_a"], ev["kelly_a"]),
    (c2, "Draw", ev["ev_d"], ev["kelly_d"]),
    (c3, team_b, ev["ev_b"], ev["kelly_b"]),
]:
    ev_sign = "+" if ev_val > 3 else "-" if ev_val < 0 else "±"
    col.markdown(
        f"<div class='ev-card'>"
        f"<div class='ev-label'>{label}</div>"
        f"<div class='ev-value'>{ev_sign} {abs(ev_val):.1f}%</div>"
        f"<div class='ev-sub'>Kelly: {kelly_val:.1f}%</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

if ev.get("recommended"):
    bet_map = {"team_a": team_a, "draw": "Draw", "team_b": team_b}
    stake_dollars = bankroll * ev["recommended_stake_pct"] / 100
    st.markdown(
        f"<div style='border:2px solid #111111; padding:0.75rem 1rem; margin-top:0.5rem; background:#111111; color:#F9F9F7;'>"
        f"<p style='margin:0; font-family:\"JetBrains Mono\",monospace; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em;'>"
        f"<strong>RECOMMENDED BET:</strong> {bet_map[ev['recommended']]} | "
        f"Stake: {ev['recommended_stake_pct']:.1f}% = <strong>${stake_dollars:.0f}</strong></p>"
        f"</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"<div style='border:1px solid #CC0000; padding:0.75rem 1rem; margin-top:0.5rem;'>"
        f"<p style='margin:0; font-family:\"JetBrains Mono\",monospace; font-size:0.75rem; color:#CC0000;'>No +EV bet found</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

if ev.get("fav_warning"):
    st.markdown(
        f"<div style='border:1px solid #111111; padding:0.5rem 1rem; margin-top:0.25rem; background:#F5F5F5;'>"
        f"<p style='margin:0; font-family:\"JetBrains Mono\",monospace; font-size:0.7rem; color:#737373;'>{ev['fav_warning']}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ── PROBABILITY MOVEMENT ─────────────────────────────────────────────────────

if len(st.session_state["prob_history"]) >= 2:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p class='section-label'>Live Tracking</p>", unsafe_allow_html=True)
    st.markdown("<h3>Probability Movement</h3>", unsafe_allow_html=True)
    hist = st.session_state["prob_history"]
    mins = [h["minute"] for h in hist]
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(
        x=mins, y=[h["a"] for h in hist], mode="lines",
        name=team_a, line=dict(color="#111111", width=3),
    ))
    fig_p.add_trace(go.Scatter(
        x=mins, y=[h["d"] for h in hist], mode="lines",
        name="Draw", line=dict(color="#737373", width=2, dash="dot"),
    ))
    fig_p.add_trace(go.Scatter(
        x=mins, y=[h["b"] for h in hist], mode="lines",
        name=team_b, line=dict(color="#A3A3A3", width=3, dash="dash"),
    ))
    fig_p.update_layout(
        title="",
        xaxis_title="Minute",
        yaxis_title="%",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=250,
        font=dict(color="#111111", family="Inter, sans-serif"),
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1),
    )
    fig_p.update_xaxes(gridcolor="#E5E5E0")
    fig_p.update_yaxes(gridcolor="#E5E5E0")
    st.plotly_chart(fig_p, use_container_width=True)


# ── FEATURE BREAKDOWN ────────────────────────────────────────────────────────

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p class='section-label'>Deep Dive</p>", unsafe_allow_html=True)
st.markdown("<h3>Feature Contribution Analysis</h3>", unsafe_allow_html=True)
features = feat_analysis.get("features", [])
if features:
    features_sorted = sorted(
        features,
        key=lambda f: abs(f.get("score_a", 0) - f.get("score_b", 0)),
        reverse=True,
    )
    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        for i, f in enumerate(features_sorted[:7]):
            fscore = f.get("score_a", 0) - f.get("score_b", 0)
            label = score_to_label(fscore)
            pct = min(1.0, abs(fscore) / 10)
            is_pos = fscore > 0
            is_neg = fscore < 0
            bar_fill_color = "#111111" if is_pos else "#CC0000" if is_neg else "#E5E5E0"
            first_letter = f['finding'][0] if f['finding'] else 'T'
            rest_text = f['finding'][1:121] if f['finding'] else ''
            dropcap = f"<span class='dropcap'>{first_letter}</span>{rest_text}..." if i == 0 else f['finding'][:120] + '...'
            st.markdown(
                f"<div style='margin-bottom:0.75rem; border-bottom:1px solid #E5E5E0; padding-bottom:0.5rem;'>"
                f"<div style='display:flex;justify-content:space-between; font-family:\"Inter\",sans-serif; font-size:0.8rem;'>"
                f"<span><strong>{f['name']}</strong> — {label}</span>"
                f"<span style='font-family:\"JetBrains Mono\",monospace; color:#737373;'>{fscore:+.1f}/10</span>"
                f"</div>"
                f"<div class='feature-bar'><div style='height:100%;width:{pct*100}%;background:{bar_fill_color};'></div></div>"
                f"<div style='font-family:\"Inter\",sans-serif; font-size:0.7rem; color:#737373; margin-top:0.2rem;'>{dropcap}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    with col_f2:
        overall = feat_analysis["net_score"]
        st.markdown(
            f"<div style='border:2px solid #111111; padding:1.5rem; text-align:center;'>"
            f"<p style='font-family:\"JetBrains Mono\",monospace; font-size:0.6rem; text-transform:uppercase; letter-spacing:0.15em; color:#737373; margin:0;'>Overall Score</p>"
            f"<div style='font-family:\"Playfair Display\",serif; font-size:2.5rem; font-weight:900; margin:0.25rem 0;'>{overall:+.1f}</div>"
            f"<p style='font-family:\"Inter\",sans-serif; font-size:0.75rem; color:#737373; margin:0;'>"
            f"Favours {team_a if overall > 0.5 else team_b if overall < -0.5 else 'Neither'}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

        top_a = [
            f for f in features_sorted
            if f.get("direction") == "A" and f.get("score_a", 0) > 1
        ][:3]
        top_b = [
            f for f in features_sorted
            if f.get("direction") == "B" and f.get("score_b", 0) > 1
        ][:3]
        if top_a:
            st.markdown(f"<p style='font-family:\"Inter\",sans-serif; font-size:0.8rem; font-weight:600; margin-top:0.75rem;'>{team_a} advantages:</p>", unsafe_allow_html=True)
            for f in top_a:
                st.markdown(f"<p style='font-family:\"JetBrains Mono\",monospace; font-size:0.7rem; margin:0.15rem 0;'>— {f['name']}</p>", unsafe_allow_html=True)
        if top_b:
            st.markdown(f"<p style='font-family:\"Inter\",sans-serif; font-size:0.8rem; font-weight:600; margin-top:0.75rem;'>{team_b} advantages:</p>", unsafe_allow_html=True)
            for f in top_b:
                st.markdown(f"<p style='font-family:\"JetBrains Mono\",monospace; font-size:0.7rem; margin:0.15rem 0;'>— {f['name']}</p>", unsafe_allow_html=True)

        draw_feat = next(
            (f for f in features if "Draw Tendency" in f.get("name", "")),
            None,
        )
        if draw_feat:
            st.markdown(f"<div style='border:1px solid #111111; padding:0.5rem; margin-top:0.75rem; font-family:\"JetBrains Mono\",monospace; font-size:0.7rem;'>{draw_feat['finding'].split('. ')[-1]}</div>", unsafe_allow_html=True)


# ── SIGNALS ──────────────────────────────────────────────────────────────────

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p class='section-label'>Context</p>", unsafe_allow_html=True)
st.markdown("<h3>Contextual Signals</h3>", unsafe_allow_html=True)
active = [s for s in sigs["signals"] if s.get("impact", 0) > 0]
if active:
    for s in active:
        icon = "▲" if s["direction"] == "advantage" else "▼" if s["direction"] == "disadvantage" else "◆"
        team_label = team_a if s["team"] == "A" else team_b
        direction_text = "Advantage" if s["direction"] == "advantage" else "Disadvantage" if s["direction"] == "disadvantage" else "Neutral"
        impact_marks = "●" * s["impact"]
        st.markdown(
            f"<div class='signal-card'>"
            f"<div style='display:flex;justify-content:space-between; font-family:\"Inter\",sans-serif; font-size:0.85rem;'>"
            f"<span>{icon} <strong>{s['name']}</strong> — {team_label} | {direction_text}</span>"
            f"<span style='color:#111111; font-family:\"JetBrains Mono\",monospace; font-size:0.7rem;'>{impact_marks}</span>"
            f"</div>"
            f"<div style='font-family:\"Inter\",sans-serif; font-size:0.75rem; color:#737373; margin-top:0.25rem;'>{s['finding']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
else:
    st.markdown("<div style='border:1px solid #E5E5E0; padding:0.75rem; font-family:\"Inter\",sans-serif; font-size:0.8rem; color:#737373;'>No strong contextual signals for this match.</div>", unsafe_allow_html=True)


# ── BASELINE STATS ───────────────────────────────────────────────────────────

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p class='section-label'>Data</p>", unsafe_allow_html=True)
st.markdown("<h3>Baseline Statistics</h3>", unsafe_allow_html=True)
col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    st.markdown(
        f"<div class='stat-card'>"
        f"<p class='stat-label'>Recent Form</p>"
        f"<p class='stat-value'>{team_a}: {form_a.get('form_str','N/A')}</p>"
        f"<p class='stat-value'>{team_b}: {form_b.get('form_str','N/A')}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )
with col_s2:
    st.markdown(
        f"<div class='stat-card'>"
        f"<p class='stat-label'>Avg Goals (last 5)</p>"
        f"<p class='stat-value'>{team_a}: {form_a.get('goals_scored_5',0):.2f} — {form_a.get('goals_conceded_5',0):.2f}</p>"
        f"<p class='stat-value'>{team_b}: {form_b.get('goals_scored_5',0):.2f} — {form_b.get('goals_conceded_5',0):.2f}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )
with col_s3:
    st.markdown(
        f"<div class='stat-card'>"
        f"<p class='stat-label'>Head-to-Head (last {h2h['total']})</p>"
        f"<p class='stat-value'>{team_a}: {h2h['a_wins']}W — {h2h['draws']}D — {h2h['b_wins']}W :{team_b}</p>"
        f"<p class='stat-sub'>ELO: {get_elo(team_a):.0f} vs {get_elo(team_b):.0f}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ── BETTING SIMULATOR DASHBOARD ──────────────────────────────────────────────

if "active_bet" in st.session_state:
    bet = st.session_state["active_bet"]
    current_prob = get_outcome_prob(probs, bet["outcome"])
    implied_key = {
        "team_a": "implied_a", "draw": "implied_d", "team_b": "implied_b",
    }.get(bet["outcome"])
    current_market_implied = market.get(implied_key, 0)
    bet_value = simulate_bet_value(bet["stake"], bet["odds"], current_prob)
    cashout_value = (
        round(current_market_implied * bet["potential_payout"], 2)
        if current_market_implied
        else bet_value["current_value"]
    )

    if "bet_value_history" not in st.session_state:
        st.session_state["bet_value_history"] = []
    if is_live and live_data.get("live"):
        st.session_state["bet_value_history"].append({
            "minute": live_data["minute"],
            "value": bet_value["current_value"],
            "pnl": bet_value["pnl"],
        })
        st.session_state["bet_value_history"] = st.session_state["bet_value_history"][-100:]

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p class='section-label'>Portfolio</p>", unsafe_allow_html=True)
    st.markdown("<h3>Virtual Bet Dashboard</h3>", unsafe_allow_html=True)

    col_b1, col_b2, col_b3, col_b4 = st.columns(4)
    col_b1.markdown(
        f"<div class='ev-card'>"
        f"<div class='ev-label'>Bet On</div>"
        f"<div class='stat-value'>{bet['outcome_label']}</div>"
        f"<div class='ev-sub'>@ {bet['odds']:.2f}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    col_b2.markdown(
        f"<div class='ev-card'>"
        f"<div class='ev-label'>Stake / Payout</div>"
        f"<div class='stat-value'>${bet['stake']:.0f}</div>"
        f"<div class='ev-sub'>→ ${bet['potential_payout']:.2f}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    col_b3.markdown(
        f"<div class='ev-card'>"
        f"<div class='ev-label'>Current Value</div>"
        f"<div class='stat-value'>${bet_value['current_value']:.2f}</div>"
        f"<div class='ev-sub'>${bet_value['pnl']:+.2f}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    col_b4.markdown(
        f"<div class='ev-card'>"
        f"<div class='ev-label'>Cash-out / Prob</div>"
        f"<div class='stat-value'>${cashout_value:.2f}</div>"
        f"<div class='ev-sub'>Model: {current_prob*100:.1f}%</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if len(st.session_state["bet_value_history"]) > 1:
        hist_b = st.session_state["bet_value_history"]
        fig_bet = go.Figure()
        fig_bet.add_trace(go.Scatter(
            x=[h["minute"] for h in hist_b],
            y=[h["value"] for h in hist_b],
            mode="lines+markers",
            name="Bet Value",
            line=dict(color="#111111", width=2),
            fill="tozeroy",
            fillcolor="rgba(17, 17, 17, 0.04)",
        ))
        fig_bet.add_hline(
            y=bet["stake"],
            line_dash="dash",
            line_color="#737373",
            annotation_text=f"Stake: ${bet['stake']:.0f}",
        )
        fig_bet.update_layout(
            title="",
            xaxis_title="Minute",
            yaxis_title="Value ($)",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=220,
            font=dict(color="#111111", family="Inter, sans-serif"),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        fig_bet.update_xaxes(gridcolor="#E5E5E0")
        fig_bet.update_yaxes(gridcolor="#E5E5E0")
        st.plotly_chart(fig_bet, use_container_width=True)


# ── LIVE EVENTS ──────────────────────────────────────────────────────────────

if is_live and live_data.get("live") and live_data.get("events"):
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p class='section-label'>Match Feed</p>", unsafe_allow_html=True)
    st.markdown("<h3>Live Match Events</h3>", unsafe_allow_html=True)
    for ev_item in reversed(live_data["events"][-10:]):
        team_label = team_a if ev_item["team"] == "A" else team_b
        icon = (
            "⚽" if "Goal" in ev_item["type"]
            else "🟥" if "Red" in ev_item.get("detail", "")
            else "🟨"
        )
        st.markdown(
            f"<div style='border-bottom:1px solid #E5E5E0; padding:0.4rem 0; font-family:\"Inter\",sans-serif; font-size:0.85rem;'>"
            f"{icon} <strong>{ev_item['minute']}'</strong> — {ev_item['detail']} "
            f"| {team_label} | {ev_item['player']}</div>",
            unsafe_allow_html=True,
        )


# ── FULL REPORT ──────────────────────────────────────────────────────────────

st.markdown("<hr>", unsafe_allow_html=True)
with st.expander("Full Prediction Report"):
    report = build_report(
        team_a, team_b, competition, f"{venue_city}, {venue_country}",
        probs, market, ev, sigs["signals"], form_a, form_b, h2h,
        live_data if is_live else None,
        features=feat_analysis.get("features", []),
    )
    st.markdown(report)
    st.download_button(
        "Download Report",
        report,
        file_name=f"{team_a}_vs_{team_b}_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
        mime="text/markdown",
    )


# ── FOOTER ───────────────────────────────────────────────────────────────────

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    f"<div style='display:flex; justify-content:space-between; align-items:center; padding:0.5rem 0;'>"
    f"<span class='edition' style='border:none; margin:0; padding:0;'>Vol. 1 | {datetime.now().strftime('%d %b %Y')} | World Cup Edition</span>"
    f"<span class='edition' style='border:none; margin:0; padding:0;'>FIFA Prediction Engine v2 — Dixon-Coles Poisson + Dynamic ELO</span>"
    f"</div>",
    unsafe_allow_html=True,
)


# ── AUTO-REFRESH ─────────────────────────────────────────────────────────────

time.sleep(30)
st.rerun()
