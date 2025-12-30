#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_ui_playwright_docker.sh [options]

Runs the UI Playwright check in Docker with a fresh Neo4j container seeded
from michael_scott_resume.json, then tears down the compose project.

Options:
  --project NAME         Compose project name (default: ui_playwright_<UTC timestamp>)
  --timeout SECONDS      UI timeout in seconds (default: 90)
  --pdf-timeout SECONDS  PDF timeout in seconds (default: 45)
  --headed              Run Playwright headed
  --slowmo MS            Slow down Playwright actions in ms
  --allow-db-error       Allow DB error banners in UI check
  --screenshot-dir DIR   Directory for Playwright screenshots/artifacts
  -h, --help             Show this help
EOF
}

project=""
ui_timeout="90"
pdf_timeout="45"
headed=0
slowmo=""
allow_db_error=0
screenshot_dir=""
neo4j_timeout="${NEO4J_TIMEOUT:-60}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project)
      project="$2"
      shift 2
      ;;
    --timeout)
      ui_timeout="$2"
      shift 2
      ;;
    --pdf-timeout)
      pdf_timeout="$2"
      shift 2
      ;;
    --headed)
      headed=1
      shift
      ;;
    --slowmo)
      slowmo="$2"
      shift 2
      ;;
    --allow-llm-error)
      echo "Error: --allow-llm-error is not supported; LLM errors must fail." >&2
      exit 2
      ;;
    --allow-db-error)
      allow_db_error=1
      shift
      ;;
    --screenshot-dir)
      screenshot_dir="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${project}" ]]; then
  project="ui_playwright_$(date -u +%Y%m%d%H%M%S)"
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "${script_dir}/.." && pwd)"
cd "${root_dir}"

compose=(docker compose -p "${project}")
PYTHON_BIN=""

cleanup() {
  "${compose[@]}" down -v >/dev/null 2>&1 || true
}
trap cleanup EXIT

ensure_python() {
  PYTHON_BIN="$(command -v python3 2>/dev/null || true)"
  if [[ -z "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="$(command -v python 2>/dev/null || true)"
  fi
  if [[ -z "${PYTHON_BIN}" ]]; then
    echo "Error: python not found on PATH." >&2
    exit 1
  fi
}

load_openai_key() {
  if [[ -n "${OPENAI_API_KEY:-}" ]]; then
    return 0
  fi
  local key_file="${OPENAI_API_KEY_FILE:-}"
  if [[ -z "${key_file}" ]]; then
    key_file="${HOME}/openaikey.txt"
  fi
  if [[ -f "${key_file}" ]]; then
    OPENAI_API_KEY="$(tr -d '\r\n' <"${key_file}")"
    export OPENAI_API_KEY
  fi
  if [[ -z "${OPENAI_API_KEY:-}" && -n "${SUDO_USER:-}" ]]; then
    local sudo_home
    sudo_home="$(getent passwd "${SUDO_USER}" | cut -d: -f6)"
    if [[ -n "${sudo_home}" && -f "${sudo_home}/openaikey.txt" ]]; then
      OPENAI_API_KEY="$(tr -d '\r\n' <"${sudo_home}/openaikey.txt")"
      export OPENAI_API_KEY
    fi
  fi
}

load_gemini_key() {
  if [[ -n "${GEMINI_API_KEY:-}" || -n "${GOOGLE_API_KEY:-}" ]]; then
    return 0
  fi
  local key_file="${GEMINI_API_KEY_FILE:-}"
  if [[ -z "${key_file}" ]]; then
    key_file="${HOME}/geminikey.txt"
  fi
  if [[ -f "${key_file}" ]]; then
    while IFS= read -r line; do
      line="$(echo "${line}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
      if [[ -z "${line}" || "${line}" == \#* ]]; then
        continue
      fi
      if [[ "${line}" == *"="* ]]; then
        key="${line%%=*}"
        val="${line#*=}"
        key="$(echo "${key}" | tr '[:lower:]' '[:upper:]' | xargs)"
        val="$(echo "${val}" | xargs)"
        if [[ "${key}" == "GEMINI_API_KEY" ]]; then
          GEMINI_API_KEY="${val}"
          export GEMINI_API_KEY
          return 0
        fi
        if [[ "${key}" == "GOOGLE_API_KEY" ]]; then
          GOOGLE_API_KEY="${val}"
          export GOOGLE_API_KEY
          return 0
        fi
        continue
      fi
      GEMINI_API_KEY="${line}"
      export GEMINI_API_KEY
      return 0
    done <"${key_file}"
  fi
  if [[ -z "${GEMINI_API_KEY:-}" && -z "${GOOGLE_API_KEY:-}" && -n "${SUDO_USER:-}" ]]; then
    local sudo_home
    sudo_home="$(getent passwd "${SUDO_USER}" | cut -d: -f6)"
    if [[ -n "${sudo_home}" && -f "${sudo_home}/geminikey.txt" ]]; then
      GEMINI_API_KEY="$(tr -d '\r\n' <"${sudo_home}/geminikey.txt")"
      export GEMINI_API_KEY
    fi
  fi
}

wait_for_neo4j() {
  local timeout="${1:-120}"
  local start_ts
  local deadline_ts
  local attempt=0
  start_ts="$(date +%s)"
  deadline_ts=$((start_ts + timeout))
  while true; do
    attempt=$((attempt + 1))
    cid="$("${compose[@]}" ps -q neo4j 2>/dev/null || true)"
    if [[ -n "${cid}" ]]; then
      status="$(docker inspect -f '{{.State.Health.Status}}' "${cid}" 2>/dev/null || true)"
      if [[ "${status}" == "healthy" ]]; then
        echo "[ui-docker] neo4j healthy after ${attempt} checks"
        return 0
      fi
    fi
    now_ts="$(date +%s)"
    if (( now_ts < deadline_ts )); then
      echo "[ui-docker] waiting for neo4j health (attempt ${attempt}, elapsed $((now_ts - start_ts))s)"
      sleep 1
    else
      return 1
    fi
  done
}

log_dir="${root_dir}/maxcov_logs"
log_file="${log_dir}/ui_playwright_check_docker.log"
mkdir -p "${log_dir}"

echo "[ui-docker] build ui-tests"
ensure_python
need_openai=0
need_gemini=0
model_hint="${LLM_MODEL:-}"
if [[ -n "${model_hint}" ]]; then
  if [[ "${model_hint}" == gemini:* ]]; then
    need_gemini=1
  elif [[ "${model_hint}" == openai:* ]]; then
    need_openai=1
  else
    need_openai=1
  fi
else
  need_gemini=1
fi
if [[ "${model_hint}" == "openai:gpt-5.2-pro" ]]; then
  need_openai=0
  need_gemini=1
fi
if (( need_openai )); then
  load_openai_key
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "Error: OPENAI_API_KEY missing (set OPENAI_API_KEY or OPENAI_API_KEY_FILE)." >&2
    exit 1
  fi
fi
if (( need_gemini )); then
  load_gemini_key
  if [[ -z "${GEMINI_API_KEY:-}" && -z "${GOOGLE_API_KEY:-}" ]]; then
    echo "Error: GEMINI_API_KEY/GOOGLE_API_KEY missing (set GEMINI_API_KEY, GOOGLE_API_KEY, or GEMINI_API_KEY_FILE)." >&2
    exit 1
  fi
fi
"${compose[@]}" build ui-tests

echo "[ui-docker] start neo4j"
"${compose[@]}" up -d neo4j
echo "[ui-docker] wait for neo4j (timeout ${neo4j_timeout}s)"
if ! wait_for_neo4j "${neo4j_timeout}"; then
  echo "Error: neo4j did not reach healthy state in time." >&2
  exit 1
fi

env_args=(
  --env "UI_TIMEOUT=${ui_timeout}"
  --env "UI_PDF_TIMEOUT=${pdf_timeout}"
)
if [[ "${headed}" -eq 1 ]]; then
  env_args+=(--env "UI_HEADED=1")
fi
if [[ -n "${slowmo}" ]]; then
  env_args+=(--env "UI_SLOWMO=${slowmo}")
fi
if [[ "${allow_db_error}" -eq 1 ]]; then
  env_args+=(--env "UI_ALLOW_DB_ERROR=1")
fi
if [[ -n "${screenshot_dir}" ]]; then
  env_args+=(--env "UI_SCREENSHOT_DIR=${screenshot_dir}")
fi

echo "[ui-docker] run ui-tests"
set +e
"${compose[@]}" run --rm "${env_args[@]}" ui-tests 2>&1 | tee "${log_file}"
rc=${PIPESTATUS[0]}
set -e
exit "${rc}"
