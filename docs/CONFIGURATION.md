# Configuration

This repo uses environment variables for provider keys, database settings, and long-run test controls. Keep secrets out of source control.

## Database

- `NEO4J_URI` (default `bolt://127.0.0.1:7687`)
- `NEO4J_USER` (default `neo4j`)
- `NEO4J_PASSWORD` (default `ResumeBuilder`)

## LLM providers

- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `GOOGLE_API_KEY`
- Optional key files: `~/openaikey.txt` (OpenAI) and `~/geminikey.txt` (Gemini).

## LLM selection and limits

- `LLM_MODEL` (e.g., `gemini:gemini-3-flash-preview` or `openai:gpt-4o-mini`)
- `LLM_MODELS` (comma-separated list)
- `LLM_REASONING_EFFORT` or `OPENAI_REASONING_EFFORT`
- `LLM_MAX_OUTPUT_TOKENS`
- `OPENAI_MAX_OUTPUT_TOKENS` / `GEMINI_MAX_OUTPUT_TOKENS`
- `LLM_MAX_OUTPUT_TOKENS_RETRY`
- `LLM_JSON_RETRY_ATTEMPTS` (retries after non-JSON output; defaults to 2, Gemini defaults to 4). Provider overrides: `OPENAI_JSON_RETRY_ATTEMPTS`, `GEMINI_JSON_RETRY_ATTEMPTS`, `GOOGLE_JSON_RETRY_ATTEMPTS`.
- `LLM_LOG_JSON_OUTPUT=1` to log LLM JSON outputs (Stage 1) for inspection.
- `LLM_JSON_LOG_DIR` to override the output directory (defaults to `/var/log/maxcov` if present, else `maxcov_logs/`).

## Coverage and maximum-coverage controls

- `REFLEX_COVERAGE=1` to enable worker coverage
- `REFLEX_COVERAGE_FILE` to specify a coverage file path
- `MAX_COVERAGE_LOG=1` for verbose max-coverage logs
- `MAX_COVERAGE_SKIP_LLM=1` to skip LLM calls
- `MAX_COVERAGE_SKIP_PDF=1` to skip PDF generation
- `MAX_COVERAGE_UI_URL` to target a specific UI URL

## UI automation

- `PLAYWRIGHT_URL`
- `REFLEX_URL`
- `REFLEX_APP_URL`

For the full set of switches, refer to the CLI help in `harness.py`.

## Resume assets

- Import a resume JSON file with `python harness.py --import-assets path/to/resume.json`.
- Use `--overwrite-resume` to replace an existing Resume node; otherwise imports refuse to overwrite.
- `michael_scott_resume.json` is the reference asset schema used for tests.
