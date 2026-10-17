# Production operations

## Production prerequisite

Daily production requires an approved model at:

```text
PREDIXLAB_DATA/models/<market>/pivot_h<horizon>_<tag>.pkl
```

Check resolved paths first:

```bash
uv run predixlab-signals info --market us --horizon 10 --tag default
```

`daily` does not train or promote models and does not fall back to candidates. Missing models, integrity failure, or code incompatibility fail the job. See [model training and promotion](model-training.md).

## Daily job

Dry-run mounted data and models before publication:

```bash
uv run predixlab-signals daily --dry-run
uv run predixlab-signals daily --end-date 2026-08-04 --dry-run
```

Then publish:

```bash
uv run predixlab-signals daily
```

The job:

1. checks whether yesterday exists in the selected market's calendar in automatic mode;
2. locks the target to yesterday or the explicit `--end-date`;
3. validates Collector completeness and freshness, then builds labels-free inference features;
4. loads and validates the production model;
5. scores the trailing panel and warms temporal event rules;
6. extracts events from the complete history window and selects target-date events;
7. writes complete-score and event CSV files;
8. publishes Pivot and AlphaRank to MongoDB unless dry-run;
9. after both database publications succeed, writes
   `campaign.daily_signals` to `CAMPAIGN_EVENTS_QUEUE`;
10. makes a best-effort job-status write and optional operator email attempt.

Raw input normally retains the last 1,000 trading days, or `PIVOT_LOOKBACK + 500` if larger, covering nested rolling windows and EWM warm-up without reading complete training history. `daily` writes no training-dataset cache.

`--end-date` is the **required target trading date**, not merely an upper bound.
It bypasses the automatic yesterday check, but never bypasses data validation.
Missing target-date data fails the run instead of falling back to an older
session. Disabling `PIVOT_REQUIRE_TRADING_DAY` also does not disable validation.

### Daily data quality gate

Before computing features or running either Pivot or AlphaRank, the job checks:

- Coverage against the unique constituents active on the target date in
  `instruments/<market>.txt`, including symbols with missing CSV files.
- Valid, finite OHLCV and restored prices, with positive prices, consistent
  OHLC ranges, and positive stock volume (benchmarks may report zero volume).
- Unique target/prior-session records, using duplicate dates captured before
  CSV cleaning and deduplication. Identical and conflicting duplicates both
  count as invalid data; earlier historical duplicates still retain the last row.
- Freshness against the **previous trading session** in the market calendar,
  not the previous calendar day. Matching all five OHLCV fields marks a row
  as suspect; an unchanged close alone does not.
- Valid, unique, and non-repeated target/prior-session benchmark data. A bad benchmark
  fails the whole run because it supplies shared model context.

`PIVOT_DATA_MAX_BAD_RATIO` defaults to `0.05`: at least 95% of the expected
universe must be fresh and usable. Missing, unreadable, invalid, repeated, and
missing-prior-session symbols share this one budget; the same stock is counted
only once. A second coverage check includes symbols lost to insufficient
history or unusable core features. Values must be in `[0, 1)`; `0` is strict.

Within the budget, suspect target rows are excluded from both models and their
daily outputs, while valid history remains available for warm-up. Above the
budget, the job fails **before loading models or writing any prediction/pick
CSV**, including in dry-run mode. Existing snapshots and published signals are
left untouched; no campaign event is enqueued. Repair/recollect the target
session before rerunning.

Logs and `jobs.results.data_validation` record the target and comparison dates,
expected/present/valid counts, combined error ratio, affected symbols, read
errors, and benchmark errors. A failed job still attempts the normal status
write and operator notification.

## Dry-run behavior

`daily --dry-run` loads the production model, performs inference, and writes daily predictions and picks CSV files, but does not publish MongoDB, enqueue a campaign event, or send email.

By contrast, `backfill-picks --dry-run` only computes and prints a summary: it neither publishes nor writes files.

## Success, skip, and failure

| Situation | Status / exit behavior |
| --- | --- |
| Eligible trading day with zero events | `success`; writes an empty event CSV |
| Automatic mode with missing/corrupt market calendar or a non-trading yesterday | `skipped`; does not build inference data |
| Missing target session, excessive bad/missing data, or unhealthy benchmark | `failed`; non-zero exit, no new snapshots or predictions |
| Data, model, scoring, or file-write error | `failed`; non-zero CLI exit |
| Event publication failure | Calculation remains, but CLI exits non-zero |
| Campaign queue publication failure | Database publication remains, but CLI exits non-zero |
| `jobs` write or email failure | Logged; does not alter successful calculation/publication |

Monitor computation, event publication, and notification separately; email alone is not sufficient monitoring.

Campaign queue publication is enabled with:

- `CAMPAIGN_EVENTS_ENABLED=true`
- `CAMPAIGN_STORAGE_CONNECTION_STRING`
- `CAMPAIGN_EVENTS_QUEUE` (default `campaign-events`)

## Daily files

```text
PREDIXLAB_DATA/predictions/<date>/<market>/<market>_pivot.csv
PREDIXLAB_DATA/picks/<date>/<market>_pivot.csv
```

`MODELS_DIR`, `PREDICTIONS_DIR`, and `PICKS_DIR` override their roots. An event CSV exists even for zero events. See [artifacts and output locations](artifacts-and-reports.md) for fields and MongoDB documents.

### API collection contract

For API Pivot-event routes to read daily results, `PIVOT_PICKS_COLLECTION` must match the API's `pivot_picks` collection. Signals writes task status to `JOBS_COLLECTION`, while the current API `/api/v1/pivot/runs/latest` endpoint reads separate `pivot_runs`. Event publication therefore does not create that run metadata; deployments needing it must synchronize or bridge the records.

## Historical replay and publication

```bash
uv run predixlab-signals backfill-picks --start 2024-01-01 --end 2025-12-31 --dry-run
uv run predixlab-signals backfill-picks --start 2024-01-01 --end 2025-12-31
```

Replay reads saved walk-forward predictions, extracts temporal events over complete history before date filtering, and therefore retains setup, window, and cooldown context. It adds forward returns and Pivot hits by default; `--no-outcomes` omits them.

Market, horizon, tag, and per-side cap come from environment variables. No-event runs exit non-zero. Non-dry-run replay upserts MongoDB directly and creates no local historical CSV.

## Run checklist

1. `info` shows correct data, calendar, constituent, and model paths.
2. Collector data meets the contract and contains the target date.
3. `daily --dry-run` succeeds with a plausible target date and scored-symbol count.
4. Both predictions and picks CSV files are at expected paths.
5. The production-model directory is read-only to the job identity.
6. MongoDB least privilege, unique indexes, and rerun upserts are verified.
7. After production, verify event publication, `jobs` status, and notification independently.

When picks are absent, distinguish a normal zero-event result, trading-day skip, calculation failure, and publication failure.

## Related documents

- [Setup, configuration, and commands](setup-and-configuration.md)
- [Artifacts and output locations](artifacts-and-reports.md)
- [Testing, security, and limitations](testing-and-limitations.md)
