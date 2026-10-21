# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-10-21

First tagged release of the current `main` tip (`c40cf89`).

### Added

- Equity turning-signal research and publishing using normalized prices and
  constituent history from `predixlab-collector`.
- Dataset building, walk-forward training and validation, and LightGBM models.
- Pivot event publishing for buy reversals, sell reversals, and trend breakouts.
- Separate AlphaRank portfolio selection that does not consume Pivot labels,
  events, or outputs.
