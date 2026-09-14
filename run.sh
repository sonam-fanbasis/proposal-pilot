#!/usr/bin/env bash
# Start ProposalPilot: backend API on :8000, and the frontend if Node is present.
#
#   ./run.sh            build the frontend once, then serve everything from :8000
#   ./run.sh --dev      run Vite on :5173 with hot reload, proxying /api to :8000
#   ./run.sh --api-only skip the frontend entirely
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$ROOT/backend"
FRONTEND="$ROOT/frontend"
VENV="$BACKEND/.venv"
MODE="${1:-}"

export PATH="$HOME/.local/bin:$PATH"

say() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m==>\033[0m %s\n' "$*" >&2; }

# --- python environment -----------------------------------------------------
# Debian and Ubuntu ship python3 without ensurepip, so `python3 -m venv` fails
# with "ensurepip is not available". Rather than telling the user to apt-install
# something (which needs sudo they may not have), fall back to bootstrapping pip
# into ~/.local and creating the environment with virtualenv, which is
# self-contained and needs no root.
if [ ! -x "$VENV/bin/python" ]; then
  say "Creating virtual environment"
  if ! python3 -m venv "$VENV" 2>/dev/null; then
    warn "python3 -m venv unavailable (no ensurepip); bootstrapping pip into ~/.local"
    if ! python3 -m pip --version >/dev/null 2>&1; then
      curl -sS https://bootstrap.pypa.io/pip/get-pip.py -o /tmp/get-pip.py
      python3 /tmp/get-pip.py --user --quiet
    fi
    python3 -m pip install --user --quiet virtualenv
    python3 -m virtualenv --quiet "$VENV"
  fi
fi

say "Installing Python dependencies"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet -r "$BACKEND/requirements.txt"

[ -f "$BACKEND/.env" ] || { cp "$BACKEND/.env.example" "$BACKEND/.env"; say "Created backend/.env"; }

# --- seed on first run ------------------------------------------------------
if [ ! -f "$BACKEND/data/proposalpilot.db" ]; then
  say "First run: indexing the sample proposal library"
  (cd "$BACKEND" && "$VENV/bin/python" seed.py) || warn "Seeding failed; the app still starts with an empty library."
fi

if grep -qE '^GEMINI_API_KEY=.+' "$BACKEND/.env" 2>/dev/null; then
  say "Gemini key found — generated answers enabled."
else
  warn "No GEMINI_API_KEY in backend/.env — running in offline mode."
  warn "Retrieval, filtering and evaluation all work; answers are extractive quotes."
fi

# --- frontend ---------------------------------------------------------------
have_node() { command -v npm >/dev/null 2>&1; }

if [ "$MODE" != "--api-only" ] && have_node; then
  [ -d "$FRONTEND/node_modules" ] || { say "Installing frontend dependencies"; (cd "$FRONTEND" && npm install --silent); }
  if [ "$MODE" = "--dev" ]; then
    say "Starting Vite dev server on http://localhost:5173"
    (cd "$FRONTEND" && npm run dev) &
    VITE_PID=$!
    trap 'kill $VITE_PID 2>/dev/null || true' EXIT
  else
    [ -d "$FRONTEND/dist" ] || { say "Building frontend"; (cd "$FRONTEND" && npm run build); }
  fi
elif [ "$MODE" != "--api-only" ]; then
  warn "npm not found — serving the API only. Install Node 18+ for the UI."
fi

say "API on http://127.0.0.1:8000  (docs at /docs)"
[ "$MODE" = "--dev" ] && say "Open http://localhost:5173"
cd "$BACKEND"
exec "$VENV/bin/python" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 ${MODE:+--reload}
