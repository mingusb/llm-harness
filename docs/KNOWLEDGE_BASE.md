# KNOWLEDGE_BASE

## Skills
- Run full validation: `python harness.py --run-all-tests` (includes maxcov, clean reflex start, UI check).
- Generate diagrams: `python scripts/generate_diagrams.py` (writes to diagrams/).

## Lessons
- Run pyupgrade on temporary copies to avoid mutating tracked files during scans.
- [deps][REQ-006] Semgrep conflicts with reflex click/rich constraints; install semgrep with `pip install --no-deps semgrep==1.146.0` and keep click/rich for reflex.
- [deps][REQ-030] Semgrep with --no-deps still needs explicit dependency pins (boltons, click-option-group, exceptiongroup, glom, mcp, opentelemetry*, peewee, wcmatch) in requirements.txt for container runs.
- [assets][REQ-011] Asset imports refuse to overwrite existing resumes unless `--overwrite-resume` is set; run-all-tests uses an ephemeral Neo4j container so user data is not touched.
- [tests][REQ-043] detect-secrets KeywordDetector is noisy for test keys/docs; disable the plugin and exclude reflex.md to keep scans focused.
- [tests][REQ-044] pydoclint defaults are strict; pass flags to disable type-hint and return/yield checks for this repo.
- [typst][REQ-048] Center bullet markers by measuring cap height (`measure("T").height`) and applying a baseline shift of `(cap_height - bullet_height) / 2` to the marker.
- [tests][REQ-077] InterpreterPoolExecutor workers must avoid module-level globals; use local imports + JSON payload/response to satisfy shareability (pydantic_core cannot load in subinterpreters).
- [process][REQ-097] Avoid logging raw PII patterns in artifacts; use redacted placeholders for verification commands and notes.
- [llm][REQ-106] Gemini output can be empty or stop-list-violating with low max tokens; increase output tokens and sanitize prompt/output when strict stop-list enforcement is required.
- [llm][REQ-108] Set `LLM_LOG_JSON_OUTPUT=1` to log Stage 1 JSON payloads to `maxcov_logs/` (or `LLM_JSON_LOG_DIR`) for inspection.
- [ui][REQ-129] Resolve backend URLs from `/env.json` in frontend JS; backend ports can shift, so fixed port offsets are unreliable.
- [ui][REQ-132] Select2 + Reflex integration is most reliable with a native `<select>` element, dataset sync, and explicit DOM `change` dispatch from Select2 callbacks.
