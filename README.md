# Roulette Predictor

Clean, fast, mobile-first web app that predicts the **next round's color and even/odd** from your training data.

## What changed in this version
- **Brand-new minimalist UI** — light theme, single accent, no glow, no splash delay, instant load
- **No lag** — lazy tab rendering, RAF batching, no sounds (which previously fired on every tap)
- **Stronger, data-driven prediction model** — replaces the old hand-tuned scoring
- **Focus**: predicts only **Color** and **Even/Odd** (per spec)
- **Walk-forward backtesting** — each sub-model's weight in the ensemble is set by how often it would have been correct on your uploaded training history. The model literally learns *from your data*, not from gambler's-fallacy heuristics

## Prediction engine
Three sub-models vote on each outcome. Each gets a weight proportional to its backtested accuracy on your data:

1. **Laplace-smoothed 1st-order Markov chain** — `P(next | current)`
2. **Variable-length n-gram lookup** — finds the longest recent sub-sequence that occurred earlier in history, and uses what followed each match (longer matches weighted more heavily by `length^1.5`)
3. **EWMA frequency** — exponentially-weighted recent counts, half-life = 25 rounds

Final probability = weighted blend. Confidence = combination of top probability + margin over the runner-up. Cap at 92%.

## Honest accuracy expectation
Real roulette spins are **independent random events**. No model can hit 100% on a fair wheel — that's a mathematical fact. What this app can do:

- If your data comes from a **biased wheel or imperfect RNG**, the n-gram and Markov sub-models will detect the bias, get high backtest weights, and beat 50%.
- If your data comes from a **truly fair source**, accuracy will hover near the baseline (~48.6% color, ~48.6% even/odd including green/zero), no matter what.

The **Stats tab → Model insight** card shows the live backtest weight of each sub-model and how many backtest rounds each one would have hit. That's the truth meter.

## Usage
1. Open `index.html` in any modern browser
2. **Add ≥ 30 rounds** of training data (Quick tap, Bulk paste, or Upload JSON). More data → stronger backtest → better weights
3. Tap **Predict next round** on the Predict tab
4. Add the actual outcome → app auto-grades the prediction
5. Live accuracy is shown in the Predict tab; weights and hit counts on the Stats tab

## Files
```
roulette-predictor/
├── index.html      Markup + UI structure
├── styles.css      Clean light theme, no animations beyond fades
├── app.js          State, storage, prediction engine, UI logic
├── manifest.json   PWA manifest
└── README.md
```

## Accepted JSON formats
```json
[5, 17, 0, 32, 21]

[{"number": 5, "color": "red"}, {"number": 17, "color": "black"}]

{"rounds": [5, 17, 0, 32, 21]}
```

## Storage
- All data persists in `localStorage` keys `rp_rounds_v2` / `rp_preds_v2`
- Old keys from the previous version are not migrated — clear or re-import

---

⚠️ For entertainment / educational use only. Real money roulette is a negative-expectation game.
