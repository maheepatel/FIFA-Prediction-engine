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

DARK_CSS = """
<style>
    /* Dark theme base */
    .stApp { background-color: #0E1117; color: #E0E0E0; }
    .main .block-container { max-width: 1400px; padding: 1rem 2rem; }
    h1, h2, h3 { color: #FFFFFF !important; font-weight: 600; }
    .st-bx { background-color: #1A1D24; border: 1px solid #2C2F36; border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; }
    div[data-testid="metric-container"] { background: #1A1D24; border: 1px solid #2C2F36; border-radius: 10px; padding: 1rem; margin: 0.25rem; }
    div[data-testid="metric-container"] label { color: #8892A4 !important; font-size: 0.85rem; }
    div[data-testid="metric-container"] div[data-testid="metric-value"] { color: #FFFFFF !important; font-size: 1.8rem !important; font-weight: 700; }
    div[data-testid="stSelectbox"] > div { background-color: #1A1D24; border: 1px solid #2C2F36; border-radius: 8px; }
    .st-bw { background-color: #1A1D24; }
    .stButton > button { border-radius: 8px; font-weight: 600; }
    .st-emotion-cache-1y4p8pa { padding: 2rem 1rem; }
    .stProgress > div > div > div { background-image: linear-gradient(90deg, #00B4D8, #0077B6); }
    footer { display: none; }
    #MainMenu { display: none; }
    header { display: none; }
    div[data-testid="stExpander"] { background: #1A1D24; border: 1px solid #2C2F36; border-radius: 10px; }
    .conf-very-high { color: #00E676; font-weight: 700; }
    .conf-high { color: #69F0AE; font-weight: 600; }
    .conf-medium { color: #FFD740; font-weight: 600; }
    .conf-low { color: #FF6E40; font-weight: 600; }
    .conf-coinflip { color: #FF5252; font-weight: 600; }
    .prob-card { text-align: center; padding: 1rem; border-radius: 12px; }
    .prob-value { font-size: 3rem; font-weight: 800; line-height: 1; }
    .prob-label { font-size: 0.9rem; color: #8892A4; margin-top: 0.5rem; }
    .predicted-score { font-size: 1.2rem; color: #B0BEC5; text-align: center; padding: 0.5rem; }
    hr { border-color: #2C2F36; margin: 1.5rem 0; }
    .stAlert { background-color: #1A1D24; border: 1px solid #2C2F36; color: #E0E0E0; border-radius: 10px; }
    .badge { display: inline-block; padding: 0.2rem 0.8rem; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
    .badge-green { background: #00C85333; color: #00E676; border: 1px solid #00C85355; }
    .badge-red { background: #FF174433; color: #FF5252; border: 1px solid #FF174455; }
    .badge-yellow { background: #FFD74033; color: #FFD740; border: 1px solid #FFD74055; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

if "matches" not in st.session_state:
    st.session_state["matches"] = get_today_matches()
if "match_idx" not in st.session_state:
    st.session_state["match_idx"] = 0
if "prob_history" not in st.session_state:
    st.session_state["prob_history"] = []


def get_match():
    ms = st.session_state["matches"]
    idx = st.session_state["match_idx"]
    if 0 <= idx < len(ms):
        return ms[idx]
    return None


# ── SIDEBAR ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        "<h1 style='font-size:1.5rem; margin-bottom:0;'>⚽ Predict</h1>"
        "<p style='color:#8892A4; font-size:0.8rem; margin-top:0;'>"
        "Dixon-Coles Poisson + Dynamic ELO + Calibrated Ensemble</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    match = get_match()
    matches = st.session_state["matches"]
    match_labels = [f"{m['home']} vs {m['away']}" for m in matches]

    team_a = "Team A"
    team_b = "Team B"
    competition = "FIFA World Cup 2026"
    venue_city = "Unknown"
    venue_country = "Unknown"
    is_live = False

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
        else:
            first = matches[0]
            team_a = first["home"]
            team_b = first["away"]
            competition = first.get("comp", competition)
            venue_city = first.get("city", venue_city)
            venue_country = first.get("country", venue_country)
            is_live = first.get("status", "SCHEDULED") in ("LIVE", "IN_PLAY", "PAUSED")
            st.session_state["match_idx"] = 0
    else:
        st.warning("No matches available.")

    st.session_state["team_a"] = team_a
    st.session_state["team_b"] = team_b

    bankroll = st.number_input("Bankroll ($)", value=1000, step=100)

    st.divider()
    st.markdown("#### Betting Simulator")
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

    st.divider()
    st.caption("Auto-refresh every 30s")
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
    if is_live and live_data.get("live"):
        ld = live_data
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:1rem;'>"
            f"<span style='background:#FF1744;width:12px;height:12px;border-radius:50%;display:inline-block;animation:pulse 1.5s infinite;'></span>"
            f"<h2 style='margin:0;'>{team_a} vs {team_b}</h2>"
            f"<span style='font-size:1.5rem;font-weight:700;color:#FF5252;'>"
            f"{ld['score_a']} — {ld['score_b']}</span>"
            f"<span style='color:#8892A4;'>{ld['minute']}'</span>"
            f"<style>@keyframes pulse {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.3; }} }}</style>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<h2 style='margin:0;'>{team_a} vs {team_b}</h2>"
            f"<p style='color:#8892A4;margin:0;'>{venue_str} | {competition}</p>",
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
                    f"<div style='text-align:right;'>"
                    f"<span style='color:#8892A4;font-size:0.8rem;'>Kick-off in</span><br>"
                    f"<span style='font-size:1.8rem;font-weight:700;color:#00B4D8;'>{hrs}h {mins}m</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        except Exception:
            pass


# ── PROBABILITY CARDS ────────────────────────────────────────────────────────

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h3>Win Probability</h3>", unsafe_allow_html=True)

conf = probs.get("confidence", "MEDIUM")
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
    color_a = "#00E676" if pct_a > 50 else "#4FC3F7" if pct_a > 35 else "#8892A4"
    st.markdown(
        f"<div class='prob-card' style='background:linear-gradient(135deg,#0D47A122,#1565C022);border:1px solid #1565C044;'>"
        f"<div class='prob-value' style='color:{color_a};'>{pct_a:.1f}%</div>"
        f"<div class='prob-label'>{team_a} Win</div>"
        f"<div style='margin-top:0.5rem;height:6px;background:#2C2F36;border-radius:3px;'>"
        f"<div style='height:100%;width:{min(pct_a,100)}%;background:linear-gradient(90deg,#1565C0,#42A5F5);border-radius:3px;'></div>"
        f"</div>"
        f"<div style='margin-top:0.3rem;display:flex;justify-content:space-between;font-size:0.7rem;color:#8892A4;'>"
        f"<span>xG: {probs.get('xg_a','-')}</span>"
        f"<span>ELO: {elo.get('elo_a',0):.0f}</span>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

with col_d:
    pct_d = probs["draw"]
    st.markdown(
        f"<div class='prob-card' style='background:linear-gradient(135deg,#F57F1722,#F57F1711);border:1px solid #F57F1744;'>"
        f"<div class='prob-value' style='color:#FFD740;'>{pct_d:.1f}%</div>"
        f"<div class='prob-label'>Draw</div>"
        f"<div style='margin-top:0.5rem;height:6px;background:#2C2F36;border-radius:3px;'>"
        f"<div style='height:100%;width:{min(pct_d,100)}%;background:linear-gradient(90deg,#F57F17,#FFD740);border-radius:3px;'></div>"
        f"</div>"
        f"<div style='margin-top:0.3rem;font-size:0.7rem;color:#8892A4;'>"
        f"<span>{probs.get('predicted_winner','?')}</span>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

with col_b:
    pct_b = probs["b_win"]
    color_b = "#FF5252" if pct_b > 50 else "#EF5350" if pct_b > 35 else "#8892A4"
    st.markdown(
        f"<div class='prob-card' style='background:linear-gradient(135deg,#B71C1C22,#C6282822);border:1px solid #C6282844;'>"
        f"<div class='prob-value' style='color:{color_b};'>{pct_b:.1f}%</div>"
        f"<div class='prob-label'>{team_b} Win</div>"
        f"<div style='margin-top:0.5rem;height:6px;background:#2C2F36;border-radius:3px;'>"
        f"<div style='height:100%;width:{min(pct_b,100)}%;background:linear-gradient(90deg,#C62828,#EF5350);border-radius:3px;'></div>"
        f"</div>"
        f"<div style='margin-top:0.3rem;display:flex;justify-content:space-between;font-size:0.7rem;color:#8892A4;'>"
        f"<span>xG: {probs.get('xg_b','-')}</span>"
        f"<span>ELO: {elo.get('elo_b',0):.0f}</span>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

conf_icons = {
    "VERY HIGH": "🟢", "HIGH": "🟢", "MEDIUM": "🟡",
    "LOW": "🟠", "COINFLIP": "🔴",
}
st.markdown(
    f"<div style='text-align:center;padding:0.5rem;'>"
    f"<span class='{conf_class}'>{conf_icons.get(conf, '⚪')} Confidence: {conf}</span>"
    f"<span style='color:#8892A4;margin-left:2rem;'>"
    f"Predicted: {probs.get('xg_a','-')}–{probs.get('xg_b','-')}</span>"
    f"<span style='color:#8892A4;margin-left:2rem;'>ELO gap: {probs.get('elo_diff',0):+.0f}</span>"
    f"</div>",
    unsafe_allow_html=True,
)


# ── MODEL vs MARKET ──────────────────────────────────────────────────────────

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h3>Model vs Market</h3>", unsafe_allow_html=True)

col_chart, col_meta = st.columns([3, 1])
with col_chart:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Our Model",
        x=[team_a, "Draw", team_b],
        y=[probs["a_win"], probs["draw"], probs["b_win"]],
        marker_color=["#42A5F5", "#FFD740", "#EF5350"],
        opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        name=f"Market ({market.get('source', 'Odds')})",
        x=[team_a, "Draw", team_b],
        y=[
            market["implied_a"] * 100,
            market["implied_d"] * 100,
            market["implied_b"] * 100,
        ],
        marker_color=["#1A237E", "#F57F17", "#B71C1C"],
        opacity=0.6,
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
            marker=dict(color="#00E676", size=14, symbol="diamond"),
        ))
    fig.update_layout(
        barmode="group",
        yaxis_title="Probability (%)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=320,
        font=dict(color="#E0E0E0"),
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", y=1.1),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#2C2F36")
    fig.update_yaxes(gridcolor="#2C2F36", range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

with col_meta:
    if poly_prices:
        st.markdown(
            f"<div style='background:#1A1D24;border:1px solid #00E67644;border-radius:10px;padding:1rem;'>"
            f"<p style='color:#00E676;font-weight:600;margin:0;'>Polymarket</p>"
            f"<p style='font-size:0.8rem;color:#8892A4;'>{poly_prices.get('market_title', '')[:40]}</p>"
            f"<p>{team_a}: <strong>{poly_prices.get('poly_a',0)*100:.1f}%</strong></p>"
            f"<p>Draw: <strong>{poly_prices.get('poly_draw',0)*100:.1f}%</strong></p>"
            f"<p>{team_b}: <strong>{poly_prices.get('poly_b',0)*100:.1f}%</strong></p>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='background:#1A1D24;border:1px solid #2C2F36;border-radius:10px;padding:1rem;text-align:center;'>"
            f"<p style='color:#8892A4;'>Polymarket data</p>"
            f"<p style='color:#8892A4;font-size:0.8rem;'>not available</p>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ── EV + KELLY ───────────────────────────────────────────────────────────────

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h3>Expected Value & Kelly Stake</h3>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
for col, label, ev_val, kelly_val in [
    (c1, team_a, ev["ev_a"], ev["kelly_a"]),
    (c2, "Draw", ev["ev_d"], ev["kelly_d"]),
    (c3, team_b, ev["ev_b"], ev["kelly_b"]),
]:
    icon = "🟢" if ev_val > 3 else "🔴" if ev_val < 0 else "🟡"
    col.markdown(
        f"<div style='background:#1A1D24;border:1px solid #2C2F36;border-radius:10px;padding:1rem;text-align:center;'>"
        f"<div style='font-size:1.2rem;font-weight:600;'>{icon} {label}</div>"
        f"<div style='font-size:2rem;font-weight:700;margin:0.25rem 0;'>"
        f"{'<span style=\"color:#00E676\">' if ev_val > 3 else '<span style=\"color:#FF5252\">' if ev_val < 0 else '<span style=\"color:#FFD740\">'}"
        f"EV: {ev_val:+.1f}%</span></div>"
        f"<div style='color:#8892A4;font-size:0.9rem;'>Kelly: {kelly_val:.1f}%</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

if ev.get("recommended"):
    bet_map = {"team_a": team_a, "draw": "Draw", "team_b": team_b}
    stake_dollars = bankroll * ev["recommended_stake_pct"] / 100
    st.success(
        f"**BET: {bet_map[ev['recommended']]}** | "
        f"Stake: {ev['recommended_stake_pct']:.1f}% = **${stake_dollars:.0f}**"
    )
else:
    st.error("No +EV bet found")

if ev.get("fav_warning"):
    st.warning(ev["fav_warning"])


# ── PROBABILITY MOVEMENT ─────────────────────────────────────────────────────

if len(st.session_state["prob_history"]) >= 2:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3>Probability Movement</h3>", unsafe_allow_html=True)
    hist = st.session_state["prob_history"]
    mins = [h["minute"] for h in hist]
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(
        x=mins, y=[h["a"] for h in hist], mode="lines",
        name=team_a, line=dict(color="#42A5F5", width=3),
    ))
    fig_p.add_trace(go.Scatter(
        x=mins, y=[h["d"] for h in hist], mode="lines",
        name="Draw", line=dict(color="#FFD740", width=2, dash="dot"),
    ))
    fig_p.add_trace(go.Scatter(
        x=mins, y=[h["b"] for h in hist], mode="lines",
        name=team_b, line=dict(color="#EF5350", width=3),
    ))
    fig_p.update_layout(
        title="",
        xaxis_title="Minute",
        yaxis_title="%",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=250,
        font=dict(color="#E0E0E0"),
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1),
    )
    fig_p.update_xaxes(gridcolor="#2C2F36")
    fig_p.update_yaxes(gridcolor="#2C2F36")
    st.plotly_chart(fig_p, use_container_width=True)


# ── FEATURE BREAKDOWN ────────────────────────────────────────────────────────

st.markdown("<hr>", unsafe_allow_html=True)
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
        for f in features_sorted[:7]:
            fscore = f.get("score_a", 0) - f.get("score_b", 0)
            icon = "🟢" if fscore > 1 else "🔴" if fscore < -1 else "⚪"
            label = score_to_label(fscore)
            pct = min(1.0, abs(fscore) / 10)
            bar_color = (
                "linear-gradient(90deg,#00E676,#69F0AE)"
                if fscore > 0
                else "linear-gradient(90deg,#FF5252,#FF8A80)"
                if fscore < 0
                else "linear-gradient(90deg,#8892A4,#B0BEC5)"
            )
            st.markdown(
                f"<div style='margin-bottom:0.75rem;'>"
                f"<div style='display:flex;justify-content:space-between;'>"
                f"<span>{icon} <strong>{f['name']}</strong> — {label}</span>"
                f"<span style='color:#8892A4;'>{fscore:+.1f}/10</span>"
                f"</div>"
                f"<div style='height:4px;background:#2C2F36;border-radius:2px;margin-top:0.25rem;'>"
                f"<div style='height:100%;width:{pct*100}%;background:{bar_color};border-radius:2px;'></div>"
                f"</div>"
                f"<div style='font-size:0.75rem;color:#8892A4;margin-top:0.2rem;'>{f['finding'][:120]}...</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    with col_f2:
        overall = feat_analysis["net_score"]
        overall_color = "#00E676" if overall > 2 else "#FF5252" if overall < -2 else "#FFD740"
        st.markdown(
            f"<div style='background:#1A1D24;border:1px solid #2C2F36;border-radius:10px;padding:1rem;text-align:center;'>"
            f"<div style='font-size:0.8rem;color:#8892A4;'>Overall Feature Score</div>"
            f"<div style='font-size:2.5rem;font-weight:700;color:{overall_color};'>{overall:+.1f}</div>"
            f"<div style='color:#8892A4;font-size:0.75rem;'>"
            f"Favours {team_a if overall > 0.5 else team_b if overall < -0.5 else 'Neither'}"
            f"</div>"
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
            st.markdown(f"**{team_a} advantages:**")
            for f in top_a:
                st.markdown(f"- {f['name']}")
        if top_b:
            st.markdown(f"**{team_b} advantages:**")
            for f in top_b:
                st.markdown(f"- {f['name']}")

        draw_feat = next(
            (f for f in features if "Draw Tendency" in f.get("name", "")),
            None,
        )
        if draw_feat:
            st.info(f"Draw Signal: {draw_feat['finding'].split('. ')[-1]}")


# ── SIGNALS ──────────────────────────────────────────────────────────────────

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h3>Contextual Signals</h3>", unsafe_allow_html=True)
active = [s for s in sigs["signals"] if s.get("impact", 0) > 0]
if active:
    for s in active:
        icon = "🟢" if s["direction"] == "advantage" else "🔴" if s["direction"] == "disadvantage" else "⚪"
        team_label = team_a if s["team"] == "A" else team_b
        direction_text = "Advantage" if s["direction"] == "advantage" else "Disadvantage" if s["direction"] == "disadvantage" else "Neutral"
        impact_stars = "⭐" * s["impact"]
        st.markdown(
            f"<div style='background:#1A1D24;border:1px solid #2C2F36;border-radius:8px;padding:0.75rem 1rem;margin-bottom:0.5rem;'>"
            f"<div style='display:flex;justify-content:space-between;'>"
            f"<span>{icon} <strong>{s['name']}</strong> — {team_label} | {direction_text}</span>"
            f"<span style='color:#FFD740;'>{impact_stars}</span>"
            f"</div>"
            f"<div style='font-size:0.8rem;color:#8892A4;margin-top:0.25rem;'>{s['finding']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
else:
    st.info("No strong contextual signals for this match.")


# ── BASELINE STATS ───────────────────────────────────────────────────────────

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h3>Baseline Statistics</h3>", unsafe_allow_html=True)
col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    st.markdown(
        f"<div style='background:#1A1D24;border:1px solid #2C2F36;border-radius:10px;padding:1rem;'>"
        f"<p style='color:#8892A4;font-size:0.8rem;margin:0;'>Recent Form</p>"
        f"<p style='font-size:1.1rem;margin:0.25rem 0;'>{team_a}: {form_a.get('form_str','N/A')}</p>"
        f"<p style='font-size:1.1rem;margin:0;'>{team_b}: {form_b.get('form_str','N/A')}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )
with col_s2:
    st.markdown(
        f"<div style='background:#1A1D24;border:1px solid #2C2F36;border-radius:10px;padding:1rem;'>"
        f"<p style='color:#8892A4;font-size:0.8rem;margin:0;'>Avg Goals (last 5)</p>"
        f"<p style='font-size:1.1rem;margin:0.25rem 0;'>{team_a}: ⚽{form_a.get('goals_scored_5',0):.2f} — 🛡️{form_a.get('goals_conceded_5',0):.2f}</p>"
        f"<p style='font-size:1.1rem;margin:0;'>{team_b}: ⚽{form_b.get('goals_scored_5',0):.2f} — 🛡️{form_b.get('goals_conceded_5',0):.2f}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )
with col_s3:
    st.markdown(
        f"<div style='background:#1A1D24;border:1px solid #2C2F36;border-radius:10px;padding:1rem;'>"
        f"<p style='color:#8892A4;font-size:0.8rem;margin:0;'>H2H (last {h2h['total']})</p>"
        f"<p style='font-size:1.1rem;margin:0.25rem 0;'>{team_a}: {h2h['a_wins']}W — {h2h['draws']}D — {h2h['b_wins']}W :{team_b}</p>"
        f"<p style='color:#8892A4;font-size:0.8rem;margin:0;'>ELO: {get_elo(team_a):.0f} vs {get_elo(team_b):.0f}</p>"
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
    st.markdown("<h3>Virtual Bet Dashboard</h3>", unsafe_allow_html=True)

    col_b1, col_b2, col_b3, col_b4 = st.columns(4)
    col_b1.markdown(
        f"<div style='background:#1A1D24;border:1px solid #2C2F36;border-radius:10px;padding:1rem;text-align:center;'>"
        f"<div style='font-size:0.8rem;color:#8892A4;'>Bet On</div>"
        f"<div style='font-size:1.3rem;font-weight:600;'>{bet['outcome_label']}</div>"
        f"<div style='font-size:0.9rem;color:#8892A4;'>@ {bet['odds']:.2f}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    col_b2.markdown(
        f"<div style='background:#1A1D24;border:1px solid #2C2F36;border-radius:10px;padding:1rem;text-align:center;'>"
        f"<div style='font-size:0.8rem;color:#8892A4;'>Stake / Payout</div>"
        f"<div style='font-size:1.3rem;font-weight:600;'>${bet['stake']:.0f}</div>"
        f"<div style='font-size:0.9rem;color:#8892A4;'>→ ${bet['potential_payout']:.2f}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    pnl_clr = '#00E676' if bet_value['pnl'] >= 0 else '#FF5252'
    col_b3.markdown(
        f"<div style='background:#1A1D24;border:1px solid #2C2F36;border-radius:10px;padding:1rem;text-align:center;'>"
        f"<div style='font-size:0.8rem;color:#8892A4;'>Current Value</div>"
        f"<div style='font-size:1.3rem;font-weight:600;'>${bet_value['current_value']:.2f}</div>"
        f"<div style='font-size:0.9rem;color:{pnl_clr};'>${bet_value['pnl']:+.2f}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    col_b4.markdown(
        f"<div style='background:#1A1D24;border:1px solid #2C2F36;border-radius:10px;padding:1rem;text-align:center;'>"
        f"<div style='font-size:0.8rem;color:#8892A4;'>Cash-out / Prob</div>"
        f"<div style='font-size:1.3rem;font-weight:600;'>${cashout_value:.2f}</div>"
        f"<div style='font-size:0.9rem;color:#8892A4;'>Model: {current_prob*100:.1f}%</div>"
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
            line=dict(color="#F39C12", width=3),
            fill="tozeroy",
            fillcolor="rgba(243, 156, 18, 0.05)",
        ))
        fig_bet.add_hline(
            y=bet["stake"],
            line_dash="dash",
            line_color="#7F8C8D",
            annotation_text=f"Stake: ${bet['stake']:.0f}",
        )
        fig_bet.update_layout(
            title="",
            xaxis_title="Minute",
            yaxis_title="Value ($)",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=220,
            font=dict(color="#E0E0E0"),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        fig_bet.update_xaxes(gridcolor="#2C2F36")
        fig_bet.update_yaxes(gridcolor="#2C2F36")
        st.plotly_chart(fig_bet, use_container_width=True)


# ── LIVE EVENTS ──────────────────────────────────────────────────────────────

if is_live and live_data.get("live") and live_data.get("events"):
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3>Live Match Events</h3>", unsafe_allow_html=True)
    for ev_item in reversed(live_data["events"][-10:]):
        team_label = team_a if ev_item["team"] == "A" else team_b
        icon = (
            "⚽" if "Goal" in ev_item["type"]
            else "🟥" if "Red" in ev_item.get("detail", "")
            else "🟨"
        )
        st.write(
            f"{icon} **{ev_item['minute']}'** — {ev_item['detail']} "
            f"| {team_label} | {ev_item['player']}"
        )


# ── FULL REPORT ──────────────────────────────────────────────────────────────

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


# ── AUTO-REFRESH ─────────────────────────────────────────────────────────────

time.sleep(30)
st.rerun()
