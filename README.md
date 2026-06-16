# ⚽ FIFA Prediction Engine v2

**Dixon-Coles Poisson + Dynamic ELO + Calibrated Ensemble**

A real-time football match prediction engine for FIFA World Cup 2026. Combines statistical models, contextual signals, and market data to produce calibrated win probabilities, expected value analysis, and betting recommendations.

## Features

- **Multi-Model Ensemble** — Poisson (60%) + ELO (40%) weighted blend with feature-based adjustments
- **Live Match Tracking** — Real-time probability updates during matches (goals, red cards, time remaining)
- **Betting Simulator** — Place virtual bets, track P&L, calculate cash-out values
- **Expected Value & Kelly Criterion** — EV analysis for all outcomes with half-Kelly stake recommendations
- **Feature Contribution Analysis** — 10+ transparent features showing what drives each prediction
- **Contextual Signals** — Altitude mismatch, rest advantage, fixture congestion, injury/suspension news, coach instability
- **Market Comparison** — Side-by-side model vs market odds vs Polymarket prediction prices
- **Live Auto-Refresh** — Updates every 30 seconds with new match data
- **Full Prediction Reports** — Downloadable markdown reports with complete analysis

## Tech Stack

| Component | Technology |
|-----------|-----------|
| UI | [Streamlit](https://streamlit.io/) |
| Charts | [Plotly](https://plotly.com/python/) |
| Models | NumPy, SciPy (Poisson distribution) |
| APIs | football-data.org, the-odds-api, api-sports.io, Polymarket CLOB |
| Data | DuckDuckGo Search, Open-Elevation, Geopy |

## Quick Start

### 1. Clone & setup

```bash
git clone https://github.com/maheepatel/FIFA-Prediction-engine.git
cd fifa-engine
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API keys (optional)

Copy `.env.example` to `.env` and add your keys:

```env
FOOTBALL_DATA_KEY=your_key_here      # https://www.football-data.org/
ODDS_API_KEY=your_key_here           # https://the-odds-api.com/
API_FOOTBALL_KEY=your_key_here       # https://www.api-football.com/
```

> Without API keys, the engine uses tier-based fallbacks — predictions still work, but with less data.

### 4. Run the app

```bash
streamlit run app.py
```

## How It Works

### Prediction Pipeline

```
Team Form (last 5 matches)
        ↓
   ELO Rating (dynamic) ──┐
                          ├── Ensemble (60/40 blend)
   Poisson (xG model)  ───┘     ↓
                          ├── Signal Adjustments (±10%)
                          ├── Feature Score (±5%)
                          ↓
               Calibrated Probabilities
                          ↓
              EV Analysis + Kelly Stake
```

### Models

1. **Dixon-Coles Poisson** — Models goals scored/conceded using attack/defense ratings normalized to league average (1.35 goals/game). Generates full score matrix up to 10 goals.

2. **Dynamic ELO** — Team ratings updated after each match with goal-difference weighting. Home advantage (+100 pts). K-factor of 32 (20 for friendlies).

3. **Ensemble** — Poisson (60%) + ELO (40%) weighted blend, adjusted by signal analysis and feature contribution scores. Platt-scaled confidence calibration.

4. **Live Update** — Remaining time Poisson projection from current score, accounting for red card advantages.

### Betting Analysis

- **Expected Value** — Model probability vs market-implied probability across all three outcomes
- **Kelly Criterion** — Half-Kelly optimal stake sizing (capped at 12% of bankroll)
- **Longshot Bias Detection** — Flags bets where market underprices low-probability outcomes
- **Virtual Bet Dashboard** — Track bet value, P&L, and cash-out amounts in real-time during live matches

## Project Structure

```
fifa-engine/
├── app.py           # Streamlit UI (all rendering)
├── engine.py        # Prediction models: Poisson, ELO, Ensemble, EV/Kelly, Live updates
├── data.py          # API calls: form, H2H, odds, live events, altitude, news, Polymarket
├── signals.py       # Contextual signal detection (altitude, rest, congestion, injury, coach)
├── features.py      # Feature contribution scoring (10+ analysis features)
├── report.py        # Markdown report builder
├── requirements.txt # Python dependencies
├── .env.example     # API key template
└── STATUS.md        # Development status and known issues
```

## API Dependencies

| API | Used For | Free Tier |
|-----|----------|-----------|
| [football-data.org](https://www.football-data.org/) | Team form, H2H, match discovery | 10 req/min |
| [the-odds-api](https://the-odds-api.com/) | Live odds from Pinnacle/Bet365 | 500 req/month |
| [api-sports.io](https://www.api-football.com/) | Live match events, red cards | 100 req/day |
| [Polymarket CLOB](https://clob.polymarket.com/) | Prediction market prices | Unlimited |
| [Open-Elevation](https://open-elevation.com/) | Venue altitude | Unlimited |
| DuckDuckGo Search | Injury & coach news | Unlimited |

## Fallback Behavior

The engine is designed to work **without any API keys**. When keys are missing, every data source falls back to tier-based estimates derived from team strength ratings:

- Form → tier-based W/D/L patterns and goals scored/conceded
- H2H → tier-differential weighted historical simulation
- Odds → tier-gap based fair odds with vig
- Live events → returns empty (model runs in pre-match mode)
- Polymarket → skipped

This means the app runs fully autonomously with zero configuration.

## License

MIT
