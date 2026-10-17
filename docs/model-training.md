# Model training and promotion

## Model output

`PivotModel` trains one shared set of LightGBM models across all eligible symbols, not one model per symbol.

| Model head | Type | Default target | Purpose |
| --- | --- | --- | --- |
| Swing position | Regression | `y_swing_pos_cs` | Estimate whether the current point is nearer the future swing low or high |
| Amplitude | Regression | `y_amplitude` | Estimate whether the expected move is large enough |
| Up/down barrier | Classification | `y_barrier_cs` | Estimate which ATR barrier is more likely to be reached first |
| Pivot low | Classification | `y_pivot_low_hold` | Estimate whether today's low persists in the following bounded window |
| Pivot high | Classification | `y_pivot_high_hold` | Estimate whether today's high persists in the following bounded window |
| Pivot low return | Regression | `y_pivot_low_return` | Estimate directional future return conditional on the low holding |
| Pivot high return | Regression | `y_pivot_high_return` | Estimate directional future return conditional on the high holding |

Barrier-timeout rows do not train the binary barrier head. `train --absolute` changes the primary targets from cross-sectionally neutral to absolute targets.

Predictions include `swing_pos_hat`, `amplitude_hat`, `p_up`, expected gains/losses, hold probabilities, conditional return estimates, and `pivot_low_score` / `pivot_high_score`. The pivot scores equal hold probability multiplied by positive expected directional return and drive turning-point gates; older bundles without return heads fall back to hold probability alone. The signed state `score` remains clipped to `[-1, 1]`: positive is bullish and negative bearish.

## Walk-forward training

```bash
uv run predixlab-signals train --market us --horizon 10 --tag default
```

## AlphaRank portfolio model

The Pivot model remains dedicated to turning-point detection. Portfolio ranking
uses an `AlphaRankEnsemble` of three `AlphaRankModel` submodels trained on the
most recent 36, 60, and 96 months. Each submodel has three heads:

- 10-session next-close to future-close excess return regression;
- same-day future-return percentile regression;
- probability of entering the future top 20%.

All model inputs are converted to same-day cross-sectional percentiles before
training. The three submodel scores and final 96-month ranking use weights
locked before fold evaluation: 25% excess-return regression, 55% future-return
percentile regression, and 20% Top20 probability. Submodel scores are
rank-averaged into a consensus gate. The rank head is primary, while the
thresholded Top20 head remains an auxiliary signal. Walk-forward folds are
used only for post-training diagnostics, never for selecting these weights.
Training dates contribute equal total sample weight even when historical
cross-section sizes differ. Dates with fewer than 100 completed forward-return
labels are excluded from all three heads. The Top20 classifier preserves the
observed daily base rate rather than class-balancing it, so its probability
output is not structurally shifted toward 50%.
Train the ensemble with:

```bash
uv run predixlab-signals train-alpha-rank \
  --market us --horizon 10 --dataset-tag default --tag default
```

Artifacts use `alpha_rank_h<horizon>_<tag>.pkl` and
`alpha_rank_walkforward_h<horizon>_<tag>.parquet`.

The matching `market/horizon/tag` dataset is built from the full historical
`PREDIXLAB_TRAINING_DATA` tree, not the clean production tree used by `daily`.
CLI defaults are three folds, 12 test months per fold, six validation months,
excluded `mkt_*` features, and cross-sectionally neutral targets.

```text
Training history → purge gap → validation period → embargo → test period
```

Each fold fits only prior training data, calibrates on validation, then predicts unseen test dates. Saved walk-forward predictions are the common out-of-sample source for `evaluate`, `backtest`, `report`, and `backfill-picks`. Use `train --help` as the authority for tunable options. `--include-market` and `--absolute` change the experiment definition and must be recorded with data and results.

## Final candidate model

After walk-forward prediction, training aggregates suitable tree counts by fold and head, refits and calibrates action thresholds on the calibration split, fits qualifying history using the selected counts, and writes a candidate bundle plus `.meta.json`.

Default LightGBM parameters are `num_leaves=31`, `max_depth=6`, `learning_rate=0.04`, and `n_estimators=400`; actual tree counts are selected per head from walk-forward results.

```text
PIVOT_ARTIFACTS/<market>/
├── reports/
│   ├── walkforward_h<horizon>_<tag>.parquet
│   └── walkforward_h<horizon>_<tag>.folds.json
└── models/
    ├── pivot_h<horizon>_<tag>.pkl
    └── pivot_h<horizon>_<tag>.meta.json
```

Although `train --help` still calls this stage “fit the production model,” its output is a candidate model and never overwrites a production model.

## Evaluation and promotion

Run at least:

```bash
uv run predixlab-signals evaluate --market us --horizon 10 --tag default
uv run predixlab-signals backtest --market us --horizon 10 --tag default
uv run predixlab-signals report --market us --horizon 10 --tag default
```

There is no promotion CLI. After approval, install the candidate at:

```text
PREDIXLAB_DATA/models/<market>/pivot_h<horizon>_<tag>.pkl
```

Copying `.meta.json` is recommended for auditability but not required to load the model: the bundle carries the authoritative integrity manifest. Production never selects the newest candidate automatically or falls back for missing or damaged models.

## Integrity and reproducibility

Bundles are Python pickles. Loading verifies embedded SHA-256 and length, which detects corruption but neither establishes trust nor makes an untrusted pickle safe. Install only models from trusted training pipelines and restrict production-model directory writes.

Record source revision; dataset Parquet and metadata; `market/horizon/tag` and non-default options; walk-forward predictions and folds; evaluation/backtest results; candidate files, metadata, and checksums; the approval decision; and the production installation location.

## Related documents

- [Dataset, features, and labels](dataset-features-labels.md)
- [Evaluation, backtesting, and reports](evaluation-backtesting.md)
- [Production operations](operations.md)
