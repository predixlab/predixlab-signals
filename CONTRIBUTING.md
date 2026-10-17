# Contributing to PredixLab Signals

Thank you for contributing. Start with [README](README.md) for the project's
scope, configuration, and operating limitations.

## Before making a change

- Use [GitHub Issues](https://github.com/predixlab/predixlab-signals/issues) for reproducible bugs and focused proposals.
  Discuss substantial interface, schema, or workflow changes before implementing them.
- Report vulnerabilities privately using [SECURITY.md](SECURITY.md), not a public issue.
- Fork the repository and work on a focused branch. Keep unrelated fixes separate.

## Development setup

Requirements: Python 3.11+ (CI uses 3.12), uv, and an OpenMP runtime for LightGBM.

From the repository root:

```bash
uv sync --frozen --extra dev --extra job
```

Follow README for optional services and local configuration. Use tracked examples
only as templates; keep real credentials and environment files outside Git.

## Validation

Run the smallest relevant checks while developing, then run the applicable project checks:

```bash
uv run pytest -q
```

Preserve causal features, historical membership, walk-forward isolation, and the research/production boundary. Model or event-rule changes need focused leakage tests and reproducible evaluation, not cherry-picked examples. Never load untrusted model pickles.

## Pull requests

CI uses Python 3.12 and uv 0.13.1 with frozen `dev` and `job` extras on main
pushes, pull requests, and manual dispatches. Image publication depends on a
successful test job. Container builds must copy all referenced package metadata,
including `LICENSE`, before `uv sync`.

- Describe the problem, intended behavior, and relevant issue.
- Follow existing project conventions; avoid unrelated refactors and dependency changes.
- Add focused regression tests for behavior changes and update public documentation in English.
- Include the commands run and their results; state any validation that could not run.
- Explain compatibility, data migration, configuration, and deployment consequences.
- Include screenshots for visible UI changes, with no customer or personal information.
- Do not include secrets, private endpoints, customer records, personal paths, or session
  profiles in source, fixtures, screenshots, logs, or generated artifacts. Review commit
  metadata and use a privacy-preserving identity if needed.

Be respectful and constructive. Harassment and disclosure of another person's
private information are not acceptable.

## Licensing

Contributions are submitted under the project's [Apache License 2.0](LICENSE).
Preserve copyright notices, third-party licenses, and source attribution. Do not
submit code, datasets, assets, or model files you do not have permission to share.
