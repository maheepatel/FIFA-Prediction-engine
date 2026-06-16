# FIFA Prediction Engine — Project Status

## Files Created (all in fifa-engine/)

| File | Purpose | Status |
|------|---------|--------|
| `requirements.txt` | Dependencies (pinned with >= for Python 3.12 compat) | ✅ Done |
| `.env` | API keys (FOOTBALL_DATA_KEY, ODDS_API_KEY, API_FOOTBALL_KEY) | ✅ Done |
| `data.py` | All API calls — team form, H2H, odds, live events, altitude, news, match discovery | ✅ Done |
| `engine.py` | Dixon-Coles Poisson, ELO, ensemble, EV/Kelly, live recalculation, betting simulator | ✅ Done |
| `signals.py` | 5 core contextual signals + 6 decorative signals (zero weight) | ✅ Done |
| `report.py` | Markdown report builder | ✅ Done |
| `app.py` | Streamlit UI — auto match discovery, live updates, betting sim, decorative signals | ✅ Done |

## Current Blockers

### ❌ Scipy / pip installation fails
- **Error:** `scipy==1.11.0` tries to compile from source because no pre-built wheel for Python 3.12 on Windows
- **Fix applied:** `requirements.txt` updated to `scipy>=1.12.0` (has Python 3.12 wheels)
- **pip not on PATH:** `python -m pip` works but `pip` command doesn't. Use `python -m pip install --user -r requirements.txt`

### ❌ Permission issues with system Python
- Python 3.12 installed at `C:\Python312\` — regular user can't write there
- **Workaround:** Use `--user` flag or create a virtual environment

## How to Resume

### Step 1 — Install dependencies
```powershell
cd fifa-engine
python -m pip install --user -r requirements.txt
```

### Step 2 — If scipy still fails, try venv
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3 — Run the app
```powershell
streamlit run app.py
```

## What the App Does (Auto Mode)

- **Auto-discovers** today's World Cup 2026 matches (Mexico vs South Africa, etc.)
- **Auto-runs** prediction without any manual input
- **Live mode:** probabilities update with each goal/red card
- **Pre-match:** shows countdown to kickoff
- **Bet simulator:** place virtual bets, watch value change in real-time
- **6 decorative signals:** captain burden, revenge motivation, social media silence, new father effect, away kit effect, media narrative trap

## Bugs Fixed During Build

| Bug | File | Fix |
|-----|------|-----|
| Poisson xG formula inverted defense | `engine.py:58-59` | Changed `attack * (1/defense)` → `attack * defense` |
| fav_warning not interpolated | `engine.py:177` | Added `f` prefix to f-string |
| Competition names mismatched dropdown | `data.py` | Added all app dropdown values to sport_map |
| Unused imports | Various | Removed `Optional`, `timezone`, `timedelta` |
| live_refresh hardcoded | `app.py` | Changed to use user's configured interval |
