# Setup, configuration, and commands

## Requirements

- Python 3.11+ (containers and CI use Python 3.12)
- `uv` (recommended) or `pip`
- An OpenMP runtime for LightGBM (`libgomp1` on Debian/Ubuntu)
- An input data tree meeting the [input data contract](data-contract.md)
- MongoDB for non-dry-run publishing
- Azure Communication Services Email only when email is enabled

The data collection component that produces the required input tree is not publicly available at this time. Signals does not collect these inputs.

For research and tests:

```bash
uv sync --extra dev
uv run predixlab-signals --help
uv run pytest -q
```

For production jobs:

```bash
uv sync --extra job
```

For development and production-job testing:

```bash
uv sync --extra dev --extra job
```

Equivalent pip development install:

```bash
python -m pip install -e ".[dev,job]"
predixlab-signals --help
```

## Configuration

Research parameters are primarily CLI options; paths and production-job settings are environment variables. The program does not load `.env` automatically—`.env.example` is only a template. To use it in a shell:

```bash
cp .env.example .env
# Edit .env, particularly data paths and credentials.
set -a
source .env
set +a
```

Never commit real credentials in `.env`.

### Paths

| Variable | Unset default | Purpose |
| --- | --- | --- |
| `PREDIXLAB_TRAINING_DATA` | `data` below the repository root | Full historical input used for dataset building and model training |
| `PREDIXLAB_DATA` | `~/.predixlab` | Clean inference input, production models, and persistent daily output |
| `PIVOT_ARTIFACTS` | `artifacts` below repository root | Rebuildable datasets, candidate models, walk-forward predictions, and research reports |

The following production override paths derive from `PREDIXLAB_DATA`:

| Variable | Derived default |
| --- | --- |
| `INDEX_DIR` | `PREDIXLAB_DATA/instruments` |
| `US_INDEX_PATH` / `CN_INDEX_PATH` | `INDEX_DIR/us.txt` / `INDEX_DIR/cn.txt` |
| `NORMALIZED_DATA_DIR` | `PREDIXLAB_DATA/normalized_data` |
| `US_NORMALIZED_DATA_DIR` / `CN_NORMALIZED_DATA_DIR` | `NORMALIZED_DATA_DIR/us_data` / `cn_data` |
| `MODELS_DIR` | `PREDIXLAB_DATA/models` |
| `PREDICTIONS_DIR` | `PREDIXLAB_DATA/predictions` |
| `PICKS_DIR` | `PREDIXLAB_DATA/picks` |

### Daily job

| Variable | Default | Meaning |
| --- | ---: | --- |
| `PIVOT_MARKET` | `us` | Lowercased market identifier |
| `PIVOT_TAG` | `default` | Dataset/model variant |
| `ALPHARANK_TAG` | `PIVOT_TAG` (default `default`) | Independently versioned AlphaRank model variant |
| `PIVOT_HORIZON` | `10` | Label and expected holding-period trading days |
| `PIVOT_LOOKBACK` | `250` | Trailing daily-score days for temporal-rule warm-up |
| `PIVOT_MAX_PER_SIDE` | `0` | Per-direction daily publication cap; `0` means no truncation |
| `PIVOT_REQUIRE_TRADING_DAY` | `true` | In automatic mode, continue only when yesterday is in the selected market's calendar; does not disable data validation |
| `PIVOT_DATA_MAX_BAD_RATIO` | `0.05` | Maximum combined fraction of missing, invalid, repeated, or unusable target-date constituents; must be in `[0, 1)` |

Booleans accept `1/0`, `true/false`, `yes/no`, and `on/off`. Invalid integers,
booleans, or data-quality ratios fail directly. Unknown `PIVOT_*` and
`ALPHARANK_*` variables are rejected to avoid silently applied typos.
`PIVOT_TRAIN_START` remains only for deployment compatibility and is ignored
because `daily` builds no training dataset. See the
[daily data quality gate](operations.md#daily-data-quality-gate) for validation
and exclusion semantics.

### Publication and notification

| Variable | Default | Meaning |
| --- | --- | --- |
| `MONGODB_URL` | `mongodb://localhost:27017` | MongoDB connection URI |
| `DATABASE_NAME` | `predixlab` | Database name |
| `PIVOT_PICKS_COLLECTION` | `pivot_picks` | Published-event collection |
| `ALPHARANK_COLLECTION` | `alpharank` | AlphaRank top-N collection |
| `ALPHARANK_GENERATIONS_COLLECTION` | `alpharank_generations` | AlphaRank session manifest collection |
| `ALPHARANK_TOP_N` | `15` | Number of deterministic AlphaRank rows per latest session |
| `JOBS_COLLECTION` | `jobs` | Job-status collection |
| `ACS_CONNECTION_STRING` | unset | ACS connection string |
| `ACS_SENDER_EMAIL` | unset | Verified sender |
| `ACS_TO_EMAILS` | unset | Comma-separated recipients |

Email is enabled only when all three ACS settings are non-empty and there is at least one recipient. Inject database and ACS credentials with a secret manager.

## Command overview

| Command | Purpose | Primary configuration |
| --- | --- | --- |
| `build-dataset` | Build features, labels, and dataset metadata | CLI |
| `train` | Walk-forward validation and candidate training | CLI |
| `evaluate` | Evaluate saved walk-forward predictions | CLI |
| `backtest` | Produce trade and portfolio simulations | CLI |
| `report` | Create an interactive event-accuracy report | CLI |
| `backfill-picks` | Replay and optionally publish historical events | Environment + date CLI options |
| `daily` | Build inference data, score, write files, and publish | Environment + `--end-date` / `--dry-run` |
| `info` | Display resolved paths and production-model location | CLI + path environment |

Use command help as the authority for options:

```bash
uv run predixlab-signals <command> --help
```

`backfill-picks` does not accept `--market`, `--horizon`, `--tag`, or `--max-per-side`; use environment variables.

## Docker

The image uses Python 3.12 slim, `libgomp1`, and the `job` extra:

```bash
docker build -t predixlab-signals:latest .
```

Compose mounts `./data` at `/data` and mounts `./data/models` read-only. Its example service runs `daily`, so data and a production model must exist before startup. For integration-only verification:

```bash
docker compose run --rm predixlab-signals python -m predixlab_signals daily --dry-run
```

Compose passes only some variables. Add ACS, tag, lookback, and other overrides explicitly in deployment configuration; `.env.example` is not injected automatically in full.

## Related documents

- [Input data contract](data-contract.md)
- [Model training and promotion](model-training.md)
- [Production operations](operations.md)
