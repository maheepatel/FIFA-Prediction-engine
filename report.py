"""
report.py — Full markdown prediction report builder.
"""
from typing import Dict, List
from datetime import datetime, timezone


def build_report(team_a: str, team_b: str, competition: str, venue: str,
                 probs: Dict, market: Dict, ev: Dict, signals: List[Dict],
                 form_a: Dict, form_b: Dict, h2h: Dict, live: Dict = None,
                 features: List[Dict] = None) -> str:

    a_bar = int(probs["a_win"] / 2)
    b_bar = int(probs["b_win"] / 2)
    d_bar = max(0, 50 - a_bar - b_bar)
    conf_icon = {"VERY HIGH": "🟢", "HIGH": "🟢", "MEDIUM": "🟡",
                 "LOW": "🟠", "COINFLIP": "🔴"}.get(
        probs.get("confidence", ""), "⚪")
    now = datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC")
    live_tag = (
        f" | 🔴 LIVE {live['minute']}' — {live['score_a']}-{live['score_b']}"
        if live and live.get("live") else ""
    )

    lines = [
        f"# Match Prediction Report{live_tag}",
        f"**{team_a} vs {team_b}** | {competition} | {venue}",
        f"Generated: {now}",
        "",
        "---",
        "",
        "## WIN PROBABILITY",
        "",
        f"```",
        f"{team_a:<22} {'█'*a_bar}{'░'*(50-a_bar)} {probs['a_win']:.1f}%",
        f"{'Draw':<22} {'█'*d_bar}{'░'*(50-d_bar)} {probs['draw']:.1f}%",
        f"{team_b:<22} {'█'*b_bar}{'░'*(50-b_bar)} {probs['b_win']:.1f}%",
        f"```",
        "",
        f"{conf_icon} Confidence: **{probs.get('confidence', '—')}**",
        f"Expected Goals: {team_a} **{probs.get('xg_a', '-')}** — {team_b} **{probs.get('xg_b', '-')}**",
        f"ELO difference: {probs.get('elo_diff', 0):+.0f} pts favoring {'neither' if abs(probs.get('elo_diff', 0)) < 5 else (team_a if probs.get('elo_diff', 0) > 0 else team_b)}",
        "",
        "---",
        "",
        "## EV + KELLY",
        "",
        "| Outcome | Our Model | Market Implied | Gap | EV | Kelly Stake |",
        "|---------|-----------|----------------|-----|----|-------------|",
        f"| {team_a} Win | {probs['a_win']:.1f}% | {market['implied_a'] * 100:.1f}% | {ev['gap_a']:+.1f}% | {ev['ev_a']:+.1f}% | {ev['kelly_a']:.1f}% |",
        f"| Draw | {probs['draw']:.1f}% | {market['implied_d'] * 100:.1f}% | {ev['gap_d']:+.1f}% | {ev['ev_d']:+.1f}% | {ev['kelly_d']:.1f}% |",
        f"| {team_b} Win | {probs['b_win']:.1f}% | {market['implied_b'] * 100:.1f}% | {ev['gap_b']:+.1f}% | {ev['ev_b']:+.1f}% | {ev['kelly_b']:.1f}% |",
        "",
        f"Market source: {market.get('source', '—')} | Overround: {market.get('overround', 0):.1f}%",
    ]

    if ev.get("recommended"):
        bet_map = {"team_a": team_a, "draw": "Draw", "team_b": team_b}
        lines.append(
            f"\n**BET: {bet_map[ev['recommended']]}** — Stake "
            f"**{ev['recommended_stake_pct']:.1f}%** of bankroll (Half-Kelly)"
        )
    else:
        lines.append("\n**No +EV bet found. Skip this match.**")

    if ev.get("fav_warning"):
        lines.append(f"\n{ev['fav_warning']}")

    from engine import get_elo
    lines += [
        "",
        "---",
        "",
        "## BASELINE STATS",
        "",
        "| | " + team_a + " | " + team_b + " |",
        "|--|---------|---------|",
        f"| Form (last 10) | {form_a.get('form_str', 'N/A')} | {form_b.get('form_str', 'N/A')} |",
        f"| Goals scored/game | {form_a.get('goals_scored_5', 0):.2f} | {form_b.get('goals_scored_5', 0):.2f} |",
        f"| Goals conceded/game | {form_a.get('goals_conceded_5', 0):.2f} | {form_b.get('goals_conceded_5', 0):.2f} |",
        f"| Matches in 21 days | {form_a.get('matches_last_21', 0)} | {form_b.get('matches_last_21', 0)} |",
        f"| ELO Rating | {get_elo(team_a):.0f} | {get_elo(team_b):.0f} |",
        "",
        f"**H2H (last {h2h['total']} meetings):** {team_a} {h2h['a_wins']}W — {h2h['draws']}D — {h2h['b_wins']}W {team_b}",
    ]

    if features:
        lines += ["", "---", "", "## FEATURE CONTRIBUTION", ""]
        sorted_feats = sorted(
            features,
            key=lambda f: abs(f.get("score_a", 0) - f.get("score_b", 0)),
            reverse=True,
        )
        for f in sorted_feats[:10]:
            fscore = f.get("score_a", 0) - f.get("score_b", 0)
            icon = "🟢" if fscore > 1 else "🔴" if fscore < -1 else "⚪"
            team_fav = team_a if fscore > 0 else team_b if fscore < 0 else "Neither"
            bar_len = int(min(20, abs(fscore) * 2))
            bar = "█" * bar_len + "░" * (20 - bar_len)
            lines.append(f"{icon} **{f['name']}** — Favours: {team_fav} | Score: {fscore:+.1f}")
            lines.append(f"  {bar}")
            lines.append(f"> {f['finding']}")
            lines.append(f"> *Source: {f['source']}*")
            lines.append("")

    lines += ["---", "", "## CONTEXTUAL SIGNALS", ""]
    active = [s for s in signals if s.get("impact", 0) > 0]
    if active:
        for s in active:
            icon = ("🟢" if s["direction"] == "advantage"
                    else "🔴" if s["direction"] == "disadvantage" else "⚪")
            team_label = team_a if s["team"] == "A" else team_b
            lines.append(
                f"{icon} **{s['name']}** — {team_label} | "
                f"Impact: {'⭐' * s['impact']}"
            )
            lines.append(f"> {s['finding']}")
            lines.append(f"> *Source: {s.get('source', '')}*")
            lines.append("")
    else:
        lines.append("No significant contextual signals for this match.")

    if live and live.get("live"):
        lines += [
            "",
            "---",
            "",
            "## LIVE MATCH DATA",
            "",
            f"**Minute:** {live['minute']}' | **Score:** {team_a} {live['score_a']} — {live['score_b']} {team_b}",
            f"Red cards: {team_a} {live['red_cards_a']} | {team_b} {live['red_cards_b']}",
            "",
            "**Recent events:**",
        ]
        for ev_item in live.get("events", [])[-5:]:
            team_label = team_a if ev_item["team"] == "A" else team_b
            lines.append(
                f"- {ev_item['minute']}' {ev_item['type']}: "
                f"{ev_item['detail']} — {team_label} ({ev_item['player']})"
            )

    lines.append(
        "\n---\n"
        "*FIFA Prediction Engine v2 | Dixon-Coles Poisson + Dynamic ELO "
        "+ Calibrated Ensemble | Model accuracy target: 65-70%*"
    )
    return "\n".join(lines)
