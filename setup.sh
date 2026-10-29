#!/usr/bin/env bash
# One-command setup for "Chat with your PDFs" (macOS / Linux).
# Safe to re-run: every step checks first and skips what's already done.
set -euo pipefail

cd "$(dirname "$0")"   # always work from the project folder, wherever it's called from

# Colours make each step easy to scan in the terminal.
ok()   { printf "\033[32m✔\033[0m %s\n" "$*"; }
info() { printf "\033[36m→\033[0m %s\n" "$*"; }
warn() { printf "\033[33m!\033[0m %s\n" "$*"; }
fail() { printf "\033[31m✘ %s\033[0m\n" "$*"; exit 1; }

# Load optional settings (model names etc.) before applying defaults.
if [ -f .env ]; then set -a; . ./.env; set +a; fi
CHAT_MODEL="${CHAT_MODEL:-llama3.2}"
EMBED_MODEL="${EMBED_MODEL:-nomic-embed-text}"
SQL_MODEL="${SQL_MODEL:-$CHAT_MODEL}"

echo
echo "  Chat with your documents & data: setup"
echo "  ───────────────────────────────────────"

# 1. Python 3.10+ (needed for the modern type-hint syntax used in the code)
PY=""
for cand in python3 python3.13 python3.12 python3.11 python3.10; do
  if command -v "$cand" >/dev/null 2>&1 &&
     "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    PY="$cand"; break
  fi
done
[ -n "$PY" ] || fail "Python 3.10+ not found. Install it from https://www.python.org/downloads/"
ok "Python $("$PY" -c 'import platform; print(platform.python_version())') ($PY)"

# 2. Ollama (runs the AI models locally, free)
command -v ollama >/dev/null 2>&1 || fail "Ollama not found. Install it from https://ollama.com/download"
ok "Ollama installed"

# 3. Virtual environment inside this folder (keeps packages isolated per project)
if [ ! -x .venv/bin/python ]; then
  info "Creating virtual environment (.venv)..."
  "$PY" -m venv .venv
fi
ok "Virtual environment ready"

# 4. Python packages
info "Installing packages (first time takes 1–3 minutes)..."
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -r requirements.txt
ok "Packages installed"

# 5. Make sure the Ollama server is running (start the Mac app if needed)
if ! ollama list >/dev/null 2>&1; then
  if [ "$(uname)" = "Darwin" ]; then
    info "Starting the Ollama app..."
    open -a Ollama || true
    for _ in $(seq 1 15); do ollama list >/dev/null 2>&1 && break; sleep 1; done
  fi
  ollama list >/dev/null 2>&1 || fail "Ollama is not running. Open the Ollama app (or run 'ollama serve') and re-run ./setup.sh"
fi
ok "Ollama is running"

# 6. Download models only if missing (they are large: ~2.3 GB total)
for model in $(printf "%s\n" "$CHAT_MODEL" "$EMBED_MODEL" "$SQL_MODEL" | awk '!seen[$0]++'); do
  if ollama list | awk 'NR>1 {print $1}' | grep -Eq "^${model}(:latest)?$"; then
    ok "Model $model already downloaded"
  else
    info "Downloading model $model..."
    ollama pull "$model"
    ok "Model $model downloaded"
  fi
done

# 7. Run the tests (a warning, not a hard stop, so you can still try the app)
info "Running tests..."
if .venv/bin/python -m pytest -q; then
  ok "All tests passed"
else
  warn "Some tests failed. The app may still run, but check the output above."
fi

echo
ok "Setup complete. Start the app with:  ./run.sh"
echo
