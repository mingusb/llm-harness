#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="$ROOT/tools/github_bootstrap_config.json"
OWNER=""
REPO=""
REMOTE_URL=""
YES=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: scripts/recreate_repo.sh [options] --yes

Deletes the GitHub repo, removes local .git metadata, recreates the repo, and
pushes a fresh main branch.

Options:
  --owner OWNER        GitHub owner/organization (optional if origin exists)
  --repo REPO          GitHub repository name (optional if origin exists)
  --config PATH        Bootstrap config (default: tools/github_bootstrap_config.json)
  --remote URL         Explicit remote URL (overrides inferred origin)
  --dry-run            Print planned actions without executing
  --yes                Required confirmation for destructive operations
  -h, --help           Show this help

Environment:
  GITHUB_PAT or GITHUB_TOKEN must be set with delete_repo scope.
EOF
}

die() {
  echo "Error: $*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

parse_owner_repo_from_url() {
  local url="$1"
  if [[ "$url" =~ github\.com[:/]+([^/]+)/([^/]+)(\.git)?$ ]]; then
    echo "${BASH_REMATCH[1]} ${BASH_REMATCH[2]%.git}"
  fi
}

while [ $# -gt 0 ]; do
  case "$1" in
    --owner)
      OWNER="${2:-}"
      shift 2
      ;;
    --repo)
      REPO="${2:-}"
      shift 2
      ;;
    --config)
      CONFIG_PATH="${2:-}"
      shift 2
      ;;
    --remote)
      REMOTE_URL="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --yes)
      YES=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
done

[ "$YES" -eq 1 ] || die "Refusing to run without --yes confirmation."
[ "$ROOT" != "/" ] || die "Refusing to operate on root filesystem."
[ -f "$ROOT/harness.py" ] || die "Run this script from the repo root."
[ -f "$ROOT/tools/github_bootstrap.py" ] || die "Missing tools/github_bootstrap.py."
[ -f "$CONFIG_PATH" ] || die "Missing config: $CONFIG_PATH"

require_cmd git
require_cmd python

if [ -z "${GITHUB_PAT:-}" ] && [ -z "${GITHUB_TOKEN:-}" ]; then
  die "GITHUB_PAT or GITHUB_TOKEN must be set."
fi

if [ -z "$REMOTE_URL" ] && git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  REMOTE_URL="$(git -C "$ROOT" remote get-url origin 2>/dev/null || true)"
fi

if [ -z "$OWNER" ] || [ -z "$REPO" ]; then
  if [ -n "$REMOTE_URL" ]; then
    read -r parsed_owner parsed_repo <<<"$(parse_owner_repo_from_url "$REMOTE_URL" || true)"
    OWNER="${OWNER:-$parsed_owner}"
    REPO="${REPO:-$parsed_repo}"
  fi
fi

if [ -z "$OWNER" ] || [ -z "$REPO" ]; then
  read -r cfg_owner cfg_repo <<EOF
$(python - "$CONFIG_PATH" <<'PY'
import json, sys
path = sys.argv[1]
data = json.load(open(path))
repo = data.get("repo", {})
print(repo.get("owner", ""), repo.get("name", ""))
PY
)
EOF
  OWNER="${OWNER:-$cfg_owner}"
  REPO="${REPO:-$cfg_repo}"
fi

[ -n "$OWNER" ] || die "Owner not resolved; pass --owner or set origin."
[ -n "$REPO" ] || die "Repo not resolved; pass --repo or set origin."

REMOTE_URL="${REMOTE_URL:-https://github.com/${OWNER}/${REPO}.git}"

git_name="$(git -C "$ROOT" config --get user.name || true)"
git_email="$(git -C "$ROOT" config --get user.email || true)"
[ -n "$git_name" ] || die "git user.name is not set."
[ -n "$git_email" ] || die "git user.email is not set."

echo "Target repo: ${OWNER}/${REPO}"
echo "Remote URL: ${REMOTE_URL}"

if [ "$DRY_RUN" -eq 1 ]; then
  cat <<EOF
Dry run: would execute
  python tools/github_bootstrap.py --config "$CONFIG_PATH" --owner "$OWNER" --repo "$REPO" --delete --yes
  rm -rf "$ROOT/.git"
  python tools/github_bootstrap.py --config "$CONFIG_PATH" --owner "$OWNER" --repo "$REPO"
  git -C "$ROOT" init
  git -C "$ROOT" add -A
  git -C "$ROOT" commit -m "Initial commit"
  git -C "$ROOT" branch -M main
  git -C "$ROOT" remote add origin "$REMOTE_URL"
  git -C "$ROOT" push -u origin main --force
EOF
  exit 0
fi

python "$ROOT/tools/github_bootstrap.py" --config "$CONFIG_PATH" --owner "$OWNER" --repo "$REPO" --delete --yes
rm -rf "$ROOT/.git"
python "$ROOT/tools/github_bootstrap.py" --config "$CONFIG_PATH" --owner "$OWNER" --repo "$REPO"
git -C "$ROOT" init
git -C "$ROOT" add -A
git -C "$ROOT" commit -m "Initial commit"
git -C "$ROOT" branch -M main
git -C "$ROOT" remote add origin "$REMOTE_URL"
git -C "$ROOT" push -u origin main --force
