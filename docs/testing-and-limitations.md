# Testing, security, and limitations

## Testing

The repository includes pytest tests:

```bash
uv run pytest -q
```

CI uses Python 3.12, installs locked `dev` and `job` extras, and runs the same command. Coverage includes data readers, features, labels, models, walk-forward splits, scoring, events, reports, backtests, daily jobs, data contracts, and reproducibility.

Key tested constraints:

- Appending future data does not change historical features, gates, or events.
- Forward labels exclude the signal day.
- The purge gap covers the label horizon.
- Feature selection excludes future, label, and evaluation fields.
- Event extraction uses trailing history, not a one-day cross section.
- Backtests fill at the next-day open.
- Same-day target/stop touches use stop-first behavior.
- A zero-event daily job succeeds and leaves an empty CSV.
- Missing or widely repeated target-session data stops both models before
  snapshot writes; isolated bad symbols are excluded within the shared budget.
- Candidate training does not install a production model.
- Bundles pass integrity checks before deserialization.

The repository has no documentation builder or Markdown link checker. Documentation changes should at least verify relative links, CLI `--help`, configuration keys, and `git diff`.

## Security

Use the repository [security policy](../SECURITY.md) to report vulnerabilities
privately. Do not include live credentials or personal information in reports.

- Model bundles contain pickle. SHA-256 verifies integrity but cannot make an untrusted pickle safe.
- Install only models from trusted training pipelines and make the production-model directory read-only to the job.
- Never put MongoDB or ACS credentials in the repository, logs, or research artifacts.
- Grant MongoDB credentials only required database and collection privileges.
- Record source revision, data metadata, bundle checksums, and approver/decision for promotion.
- `.env.example` is a placeholder template, not a production-secret source.

## Method limitations

- Daily OHLC cannot reveal intraday paths; backtests use conservative stop-first handling.
- Constituent-history quality depends on instrument files; `--all-history` explicitly introduces selection bias.
- Future-hold and zigzag labels describe price geometry, not necessarily profitable trades.
- Event-gate thresholds can become ineffective as market structure changes.
- Cross-sectional calibration depends on the symbol set actually available that day.
- `--include-market` materially changes sample independence and overfitting risk.
- Backtests omit capacity, impact, borrow availability, taxes, and actual order execution.
- Historical-replay returns are not a complete strategy backtest: they omit slots, targets/stops, and costs.

## Product and operational limits

- Version `0.1.0` and its interfaces/storage formats remain pre-1.0.
- Production-model promotion is a manual process outside the CLI.
- Daily data validation relies on the Collector calendar and membership
  metadata. OHLCV freshness checks detect repeated bars, not every possible
  vendor error or a guarantee that live prices are economically correct.
- `backfill-picks` receives market, horizon, and tag only through environment variables.
- Published events are not a target portfolio and do not establish capacity or borrow availability.
- MongoDB publication failure exits non-zero but does not roll back CSV files.
- `jobs` status and email are best effort and are not sufficient monitoring.
- Output supports research and engineering validation only; it is not investment advice.

## Change-validation principles

When data or signal semantics change:

1. State the information available for every row at that time.
2. Add or update focused tests for the behavioral constraint.
3. Rebuild walk-forward predictions rather than evaluating only a final model.
4. Compare fold/date stability, calibration, event lift, turnover, and after-cost results.
5. Record changes to constituents, market features, targets, or event definitions.
6. Generate and evaluate a new candidate before promotion.

## Related documents

- [Dataset, features, and labels](dataset-features-labels.md)
- [Evaluation, backtesting, and reports](evaluation-backtesting.md)
- [Production operations](operations.md)
