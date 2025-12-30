#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="/var/log/maxcov"
UI_LOG_FILE="${LOG_DIR}/ui_playwright_check.log"
REFLEX_LOG_FILE="${LOG_DIR}/reflex_ui.log"

mkdir -p "${LOG_DIR}"

UI_TIMEOUT="${UI_TIMEOUT:-360}"
UI_PDF_TIMEOUT="${UI_PDF_TIMEOUT:-45}"
UI_HEADED="${UI_HEADED:-0}"
UI_SLOWMO="${UI_SLOWMO:-}"
UI_ALLOW_DB_ERROR="${UI_ALLOW_DB_ERROR:-0}"
UI_SCREENSHOT_DIR="${UI_SCREENSHOT_DIR:-}"
UI_APP_TIMEOUT="${UI_APP_TIMEOUT:-60}"
UI_MAX_RUNTIME="${UI_MAX_RUNTIME:-585}"
LLM_LOG_JSON_OUTPUT="${LLM_LOG_JSON_OUTPUT:-1}"
export LLM_LOG_JSON_OUTPUT

START_SECONDS=${SECONDS}

check_deadline() {
  local elapsed=$((SECONDS - START_SECONDS))
  if (( elapsed >= UI_MAX_RUNTIME )); then
    echo "Error: UI tests exceeded ${UI_MAX_RUNTIME}s (elapsed=${elapsed}s)." >&2
    exit 1
  fi
}

remaining_time() {
  local remaining=$((UI_MAX_RUNTIME - (SECONDS - START_SECONDS)))
  if (( remaining < 0 )); then
    remaining=0
  fi
  echo "${remaining}"
}

if [[ "${UI_ALLOW_LLM_ERROR:-0}" -ne 0 ]]; then
  echo "Error: UI_ALLOW_LLM_ERROR is not supported; LLM errors must fail." >&2
  exit 2
fi

DEFAULT_GEMINI_MODEL="gemini:gemini-3-pro-preview"
if [[ -z "${LLM_MODEL:-}" || "${LLM_MODEL:-}" == "openai:gpt-5.2-pro" ]]; then
  export LLM_MODEL="${DEFAULT_GEMINI_MODEL}"
fi
if [[ "${OPENAI_MODEL:-}" == "gpt-5.2-pro" ]]; then
  export OPENAI_MODEL="gpt-4o-mini"
fi
echo "[ui-tests] LLM_MODEL=${LLM_MODEL} OPENAI_MODEL=${OPENAI_MODEL:-}"
if [[ "${LLM_MODEL}" == gemini:* ]]; then
  if [[ -z "${GEMINI_API_KEY:-}" && -z "${GOOGLE_API_KEY:-}" ]]; then
    echo "Error: GEMINI_API_KEY/GOOGLE_API_KEY missing (required for UI LLM pipeline)." >&2
    exit 1
  fi
else
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "Error: OPENAI_API_KEY missing (required for UI LLM pipeline)." >&2
    exit 1
  fi
fi

reflex_pid=""

stop_reflex() {
  if [[ -n "${reflex_pid}" ]] && kill -0 "${reflex_pid}" 2>/dev/null; then
    kill -INT "${reflex_pid}" 2>/dev/null || true
    for _ in {1..20}; do
      if ! kill -0 "${reflex_pid}" 2>/dev/null; then
        break
      fi
      sleep 0.25
    done
    if kill -0 "${reflex_pid}" 2>/dev/null; then
      kill -TERM "${reflex_pid}" 2>/dev/null || true
    fi
  fi
}

trap stop_reflex EXIT

wait_for_url() {
  local url="$1"
  local timeout="${2:-120}"
  echo "[ui-tests] wait for app ${url} (timeout ${timeout}s)"
  python - <<PY
import time
import urllib.request
import sys

deadline = time.time() + ${timeout}
attempt = 0
while time.time() < deadline:
    attempt += 1
    try:
        with urllib.request.urlopen("${url}", timeout=2):
            print(f"[ui-tests] app ready after {attempt} attempts")
            sys.exit(0)
    except Exception as exc:
        print(f"[ui-tests] wait app attempt {attempt} failed: {type(exc).__name__}")
        time.sleep(1)
sys.exit(1)
PY
}

echo "[ui-tests] reset neo4j"
python harness.py --reset-db michael_scott_resume.json
check_deadline

echo "[ui-tests] start app"
reflex run --frontend-port 3000 --backend-port 8000 >"${REFLEX_LOG_FILE}" 2>&1 &
reflex_pid=$!

check_deadline
remaining="$(remaining_time)"
app_timeout="${UI_APP_TIMEOUT}"
if (( app_timeout > remaining )); then
  app_timeout="${remaining}"
fi
if ! wait_for_url "http://localhost:3000" "${app_timeout}"; then
  echo "Error: app did not become ready in time." >&2
  exit 1
fi
check_deadline

ui_cmd=(
  python scripts/ui_playwright_check.py
  --url http://localhost:3000
  --timeout "${UI_TIMEOUT}"
  --pdf-timeout "${UI_PDF_TIMEOUT}"
)
if [[ "${UI_HEADED}" -eq 1 ]]; then
  ui_cmd+=(--headed)
fi
if [[ -n "${UI_SLOWMO}" ]]; then
  ui_cmd+=(--slowmo "${UI_SLOWMO}")
fi
if [[ "${UI_ALLOW_DB_ERROR}" -eq 1 ]]; then
  ui_cmd+=(--allow-db-error)
fi
if [[ -n "${UI_SCREENSHOT_DIR}" ]]; then
  ui_cmd+=(--screenshot-dir "${UI_SCREENSHOT_DIR}")
fi

echo "[ui-tests] run ui-playwright-check"
set +e
remaining="$(remaining_time)"
if (( remaining <= 0 )); then
  echo "Error: UI tests exceeded ${UI_MAX_RUNTIME}s before Playwright run." >&2
  exit 1
fi
timeout_cmd=()
if command -v timeout >/dev/null 2>&1; then
  timeout_cmd=(timeout "${remaining}")
fi
"${timeout_cmd[@]}" "${ui_cmd[@]}" 2>&1 | tee "${UI_LOG_FILE}"
rc=${PIPESTATUS[0]:-0}
set -e
if [[ "${timeout_cmd[0]:-}" == "timeout" && "${rc}" -eq 124 ]]; then
  echo "Error: UI check exceeded ${UI_MAX_RUNTIME}s overall runtime." >&2
  exit 1
fi
exit "${rc}"
