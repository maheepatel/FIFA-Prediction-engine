# FIFA Prediction Engine — Master Ideation Document
> Final version. Math-backed. Prediction market ready.

---

## VISION
A football match prediction engine that finds **positive Expected Value (+EV)** gaps
between our model's probability and the market's implied probability.

Target: **63–67% match outcome accuracy** — matching the world's best.
Real edge: Finding the 2–8% EV gaps bookmakers and prediction markets misprice.
Primary focus: FIFA World Cup + major international tournaments → club football V2.

---

## THE CORE PHILOSOPHY (from research)

**This is not a "pick winners" machine. It is an uncertainty engine.**

The pipeline is:
1. Turn data into calibrated probabilities
2. Turn probabilities into Expected Value (EV)
3. Turn EV into position sizes using Kelly Criterion
4. Only bet/stake when EV is positive

The three math laws that govern everything:
- **Dixon-Coles Poisson** → best scoreline probability model
- **Expected Value** → only metric that matters for long-term profit
- **Kelly Criterion** → optimal stake sizing, never overbet

---

## THE MATH FOUNDATION

### 1. Expected Value (EV) — The Only Bet That Matters
```
EV = (Our Probability × Potential Profit) − (Our Probability of Loss × Stake)

Example:
- Our model: Team A wins with 58% probability
- Bookmaker/Polymarket implied: 48% (odds = 2.08)
- Stake: $100
- EV = (0.58 × $108) − (0.42 × $100) = $62.64 − $42 = +$20.64

→ +EV bet. Place it. Repeat 1000 times = sustained profit.
```

**Never bet when EV ≤ 0. The edge is everything.**

### 2. Kelly Criterion — Optimal Stake Sizing
```
f* = (b×p − q) / b

Where:
  b = decimal odds − 1  (net odds)
  p = our estimated win probability
  q = 1 − p

Example:
  Our prob = 58%, odds = 2.08 (b = 1.08)
  f* = (1.08 × 0.58 − 0.42) / 1.08 = (0.6264 − 0.42) / 1.08 = 0.191

→ Bet 19.1% of bankroll
→ In practice use HALF-KELLY (9.5%) to reduce variance

Rule: If Kelly says 0 or negative → no bet.
```

### 3. Finding the Edge: Model vs Market Gap
```
Our Probability  >  Bookmaker Implied Probability  →  +EV → BET
Our Probability  <  Bookmaker Implied Probability  →  −EV → SKIP
Gap ≥ 5%  →  Strong signal, bet with confidence
Gap 2–5%  →  Weak signal, small Kelly stake
Gap < 2%  →  Noise, skip
```

### 4. Bookmaker Implied Probability (remove the vig)
```
Raw implied prob = 1 / decimal_odds
Remove vig: normalize each outcome's raw prob by total sum
Example: Home 1.90, Draw 3.40, Away 4.20
  Raw: 0.526, 0.294, 0.238 → sum = 1.058 (5.8% overround/vig)
  Normalized: 0.497, 0.278, 0.225
```

### 5. Favourite-Longshot Bias (from academic research)
```
Most exploitable range on prediction markets (Polymarket/Kalshi):
  YES contracts 5–15%  → systematically OVERPRICED → buy NO
  YES contracts 75–92% → systematically UNDERPRICED → buy YES
Sweet spot for football: strong favourite (65–85%) is usually underpriced.
Underdogs below 15% are usually overpriced → fade them.
```

---

## THE STATISTICAL CORE (60% of prediction weight)

| Model | Weight | Why |
|-------|--------|-----|
| Dixon-Coles Poisson | 25% | Best scoreline model, corrects low-score bias |
| XGBoost (calibrated) | 25% | Best ML classifier, ~67% accuracy |
| Elo / Pi-Rating | 10% | Best dynamic team quality tracker |
| Bookmaker odds (vig-removed) | 0%* | Used as reference, not weighted in model |

*Bookmaker odds are used to CALCULATE EV AFTER the model runs, not as input to the model.
If we include odds in the model, we can't measure gap vs market.

### Features that actually move the needle (from LSE/arxiv research):
1. **xG differential** (xG scored − xG conceded, last 5 games) — strongest single predictor
2. **Shots on target ratio** — more predictive than goals scored
3. **GAP Ratings** (Generalised Attacking Performance) — separate home/away attack + defense
4. **ELO difference** — best baseline, simple and proven
5. **Recent form** (exponentially weighted — recent matches 3× more important)
6. **H2H record** (last 5-6 meetings, weighted by recency)
7. **Home/away advantage** (league-specific, +0.3–0.5 xG for home teams)
8. **Key player absences** (starter-level injuries)
9. **Rest days differential** (proven fatigue effect)
10. **Fixture congestion** (matches in last 21 days)

---

## CONTEXTUAL SIGNALS (25% of prediction weight)
*Science-backed situational factors. Each has a verifiable source.*

| Signal | Source | Impact |
|--------|--------|--------|
| Altitude mismatch (training vs venue, >700m) | FIFA Medical Committee | High |
| Travel distance (>2000km = fatigue) | Calleja-González sports science | Medium-High |
| Rest days gap (3+ day differential) | UEFA workload research | Medium-High |
| Fixture congestion (>4 games in 21 days) | UEFA injury study | High |
| Vote of confidence from board | Historical football management data | High |
| Coach press conference NLP sentiment | Verifiable from news sources | Low-Medium |
| Tournament debut effect (first major tournament) | Academic tournament data | Medium |
| Squad average age (29.5+ = cliff, 24-26 = peak) | FIFA/sports science | Medium |
| Climate mismatch (heat vs cold-adapted team) | FIFA environmental research | Medium |
| Coaching staff turnover (>2 changes in 6mo) | Historical club data | Medium |

---

## SOFT INTELLIGENCE (15% of prediction weight)
*From verified news, social, official communications only. Low individual weight.*
*Only significant when 3+ soft signals align in same direction.*

- Camp morale reports (credible named journalists only)
- Active contract dispute (key player, recently reported)
- Political/national backdrop on squad (documented cases like Iran WC2022)
- Training camp location and conditions vs venue
- Player absence in training photos (with named source)

---

## DOWNGRADED SIGNALS (Decorative only — shown in dossier, zero probability weight)

These are interesting but lack statistical backing for prediction:

| Signal | Why Downgraded |
|--------|---------------|
| Captain's Burden Index | Anecdotal, inconsistent across samples |
| Revenge Motivation | Unquantifiable, case-dependent |
| Social Media Silence | Too much noise, innocent explanations |
| New Father Effect | Very small sample size |
| Away Kit Effect | No peer-reviewed evidence |
| Media Narrative Trap | Interesting theory, no causal data |

Shown in "Analyst Observations" section with label: **[ZERO BETTING WEIGHT — Analyst Colour Only]**

---

## PREDICTION MARKET STRATEGY LAYER
*(How to translate our probabilities into actual stakes on Polymarket/Kalshi)*

### Step 1: Calculate Our Probability
Run the full model → get Team A win = 61%, Draw = 22%, Team B win = 17%

### Step 2: Get Market Implied Probability
Fetch Polymarket/Kalshi current odds → remove vig → get implied probs

### Step 3: Calculate EV for Each Outcome
```python
ev = (our_prob - market_implied_prob) * (1 / market_implied_prob - 1)
# Positive = bet. Negative = skip.
```

### Step 4: Apply Favourite-Longshot Bias Filter
- If market says <15% → likely overpriced → buy NO (fade it)
- If market says 65–85% → likely underpriced → buy YES
- Never bet longshots just because they have high EV in our model — the bias is real

### Step 5: Kelly Sizing
```python
f = (b * p - q) / b  # Full Kelly
stake = bankroll * f * 0.5  # Half-Kelly (safer)
# Never bet more than 10% of bankroll on any single match
```

### Step 6: Minimum Threshold
- Only execute if EV > 3% (removes noise trades)
- Only execute if our model confidence is MEDIUM or above
- Never execute on LOW confidence calls

---

## INFORMED FLOW SIGNALS (Prediction Market Specific)
*From Ottaviani & Sorensen research — late money is smart money*

- Track Polymarket contract prices in the last 60 minutes before kickoff
- A 5%+ price shift with above-average volume = informed participant entering
- If sharp late movement aligns with our model → confidence UP
- If sharp late movement contradicts our model → re-evaluate, reduce stake

---

## OUTPUT FORMAT

### Report 1 — 72 Hours Before Match
Full signal scan. Statistical model run. EV calculated vs current market odds.

### Report 2 — 24 Hours Before Match
Updated scan. **Highest-weighted report.**
Delta vs 72hr. Final lineup signals. Updated EV if odds moved.

### Visual Output Structure
1. **Signal Heatmap** — Both teams, all categories, Green/Red/Grey
2. **Win Probability Bar** — Our model vs Bookmaker implied, side by side
3. **EV Display** — Explicit +EV or −EV with % for each outcome
4. **Kelly Stake Recommendation** — "Bet X% of bankroll on [outcome]" or "No bet"
5. **Verdict Card** — Winner, confidence, top 3 signals, sources
6. **Full Dossier** — Every signal scored, sourced, weight tier shown
7. **Analyst Observations** — Decorative signals, clearly labelled zero weight

---

## DATA SOURCES

**Stats:** football-data.org (free), OpenLigaDB (free), StatsBomb open data
**Odds/Market:** The Odds API (free tier), Polymarket API, Kalshi API
**News:** DuckDuckGo search (free, no key), direct article scraping
**Altitude/Geo:** Open-Elevation API (free)
**NLP:** HuggingFace cardiffnlp/twitter-roberta-base-sentiment (free, local)
**LLM:** Ollama local (free) / Groq free tier

❌ No proxies. No fabrication. Every signal needs a URL or named source.

---

## REALISTIC ACCURACY TARGETS

| Model | Accuracy |
|-------|----------|
| Naive baseline (always pick home) | ~46% |
| Dixon-Coles Poisson alone | ~60% |
| Poisson + ELO ensemble | ~62% |
| XGBoost + domain features | ~63–67% |
| Our target | **63–67%** |
| Bookmaker implied | ~65% |
| Theoretical max (chaos floor) | ~70–72% |

**The real win is not just accuracy. It's finding +EV gaps.**
A model that's right 58% of the time but consistently finds markets that price it at 48%
will make more money than a 67% accurate model that matches the market perfectly.


---

## REAL-WORLD MODEL FEATURES (from deployed 63-67% systems)

Research across arxiv, LSE, Springer, and Frontiers journals confirms these features
are what separates 60% models from 67% models. Add all of these to the XGBoost feature set:

### Tier 1 — Strongest predictors (must have)
| Feature | Notes |
|---------|-------|
| xG differential (last 5 games) | Strongest single predictor per LSE research |
| Shots on target ratio | More stable than goals — less luck variance |
| GAP ratings (home/away attack + defense, separate) | Wheatcroft/LSE method, outperforms single rating |
| ELO / Pi-rating difference | Best dynamic quality baseline |
| Exponentially weighted recent form | Last match weighted 3× more than 5 matches ago |
| Rolling average goals scored/conceded (last 3 and 10) | Both windows, not just one |
| Home/away split statistics (separate, not combined) | Teams behave differently home vs away |
| Key player absence indicator (top 5 by rating) | Binary per player, not just count |

### Tier 2 — Strong predictors (should have)
| Feature | Notes |
|---------|-------|
| Possession % rolling average | Direct from match stats |
| Shots total + shots on target (separate) | Both matter independently |
| Corners won (rolling avg) | Attacking pressure proxy |
| Fouls committed (rolling avg) | Tactical discipline signal |
| Yellow/red cards (rolling avg) | Discipline + suspension risk |
| Formation used (one-hot encoded) | Tactical matchup matters |
| Referee assignment (home win rate for that referee) | Documented bias effect |
| Attendance / crowd size (normalized) | Larger crowd = more home advantage |
| Stage of competition (group/knockout/final) | Risk-taking behavior changes |
| Competition tier weight | World Cup ≠ friendly |

### Tier 3 — Supplementary (nice to have)
| Feature | Notes |
|---------|-------|
| Squad market value ratio | Proxy for talent gap |
| Tactical compactness score (if data available) | Block positioning matters |
| Manager win rate vs specific formations | Tactical history |
| Days since last managerial change | System disruption window |
| Pre-match odds movement (line movement %) | Market correction signal |

---

## BACKTESTING FRAMEWORK

**This is what separates a demo from a credible product.**

Before trusting any prediction, the model must be validated against historical data
it has never seen. Without backtesting, accuracy claims are meaningless.

### Backtesting Protocol
```
1. Split historical data: 70% train / 15% validate / 15% holdout test
2. Use TimeSeriesSplit — never leak future data into training
3. Evaluate on holdout test set only (never re-tune on it)
4. Metrics to track:
   - Match outcome accuracy (target: 63-67%)
   - Brier score (probability calibration — lower = better)
   - Log loss (penalizes overconfident wrong predictions)
   - ROI % if Kelly betting was applied historically
   - EV hit rate (% of +EV bets that were actually correct)
5. Run Platt scaling or isotonic regression to calibrate raw probabilities
6. Re-validate every 3 months with new match data
```

### Signal Backtesting (per individual signal)
```
For each contextual signal (altitude, travel, rest, congestion, etc.):
1. Find all historical matches where signal fired
2. Check actual outcome vs prediction
3. Record: times_fired, times_correct, accuracy %
4. If signal accuracy < 52% over 50+ samples → reduce its weight
5. If signal accuracy > 65% over 50+ samples → increase its weight
```

This is stored in the `signal_accuracy` database table and auto-updates after each match result.

---

## QUALITY SCORES (Target after backtesting validates the model)

| Dimension | Current | Target after backtesting |
|-----------|---------|--------------------------|
| Interesting demo / analyst tool | 9/10 | 9/10 |
| Prediction accuracy | 7/10 | 7.5/10 |
| Scientific rigor | 7.5/10 | 8/10 |
| Investor / B2B attractiveness | 7/10 | 8/10 |

**How to get there:**
- Interesting demo → Already there. Heatmap + EV card + Kelly recommendation is visually compelling.
- Prediction accuracy → Backtesting confirms 63-67%. Calibration (Platt/isotonic) gets us there.
- Scientific rigor → Every signal has a cited source. Backtesting adds statistical validation.
- Investor attractiveness → Track record of correct +EV calls over 50+ matches = the proof of concept.

