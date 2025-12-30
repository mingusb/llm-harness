# Testing

This repo is designed to validate long-horizon behavior. Testing is built around coverage-driven simulations and UI automation.

Tests are optional and must be explicitly requested. Do not run test commands by default. Before any run, confirm the
Neo4j target is non-production (stub or ephemeral). If you cannot verify the target, stop and ask. When tests run in
the dockerized/ephemeral flows (`--run-all-tests` or `--run-all-tests-local`), no separate Neo4j backup is required.

## Core paths

```bash
# End-to-end maximum-coverage simulation (skips LLM calls by default)
python harness.py --maximum-coverage

# Run the full test gate (dockerized; uses scripts/run_maxcov_e2e.sh --force)
python harness.py --run-all-tests

# Run the local test pipeline (intended for container usage)
python harness.py --run-all-tests-local

# Run the Reflex coverage server and drive it via Playwright
python harness.py --maximum-coverage-reflex

# UI traversal check (requires a running Reflex app)
python harness.py --ui-playwright-check

# Docker-only UI Playwright check (spins up Neo4j + app, seeds data)
python harness.py --ui-playwright-check-docker

# Docker-only UI test gate (Neo4j + UI only)
python harness.py --run-ui-tests
```

## Docker-based runs

```bash
# Runs the max coverage mode in a container
./scripts/run_maxcov.sh
```

Note: `--run-all-tests` invokes `scripts/run_maxcov_e2e.sh --force`, which resets
all Docker containers/images/volumes/networks and uses `expect` to prompt for the
sudo password. Avoid running it on shared machines. `scripts/` is actively
evolving for UI coverage and long-run checks, so avoid changing those files while
tests are running.

## Coverage notes

- `--maximum-coverage` exercises UI state transitions, DB paths, and PDF branches.
- `REFLEX_COVERAGE=1` enables per-worker coverage tracking.
- `MAX_COVERAGE_LOG=1` emits detailed progress logs (useful for long runs).
- `--run-all-tests-local` runs against the containerized Neo4j service; treat it as ephemeral and
  ensure it is seeded with `michael_scott_resume.json` before UI checks.

## Failure path simulation

The maximum-coverage mode supports failure simulation for DB, LLM, and PDF paths. See the CLI help in `harness.py` for toggles such as:

- `--maximum-coverage-failures`
- `MAX_COVERAGE_FORCE_DB_ERROR_ON_LOAD=1`
- `MAX_COVERAGE_SKIP_PDF=1`

## UI automation

The Playwright checks validate the UI and PDF embed surface. Use these environment variables as needed:

- `PLAYWRIGHT_URL`
- `REFLEX_URL`
- `REFLEX_APP_URL`

## Suggested long-run cadence

For multi-hour or multi-day runs (only when explicitly requested):

1. Run a clean `--maximum-coverage` baseline.
2. Execute the agent objective.
3. Repeat `--maximum-coverage` every N hours or after major changes.
4. Run UI checks for regression coverage when the UI surface changes.
