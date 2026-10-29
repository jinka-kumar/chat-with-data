#!/usr/bin/env bash
# Start the app. Extra arguments go to Streamlit, e.g.  ./run.sh --server.port 8502
set -euo pipefail

cd "$(dirname "$0")"

# Settings from .env (CHAT_MODEL, EMBED_MODEL, DB_PATH) are passed to the app as env vars.
if [ -f .env ]; then set -a; . ./.env; set +a; fi

[ -x .venv/bin/python ] || { echo "Not set up yet. Run ./setup.sh first."; exit 1; }

# The app needs the Ollama server; start the Mac app automatically if it's closed.
if ! ollama list >/dev/null 2>&1; then
  if [ "$(uname)" = "Darwin" ]; then
    echo "Starting the Ollama app..."
    open -a Ollama || true
    for _ in $(seq 1 15); do ollama list >/dev/null 2>&1 && break; sleep 1; done
  fi
  ollama list >/dev/null 2>&1 || { echo "Ollama is not running. Open the Ollama app and try again."; exit 1; }
fi

# exec = Streamlit replaces this script, so Ctrl+C stops it cleanly.
exec .venv/bin/python -m streamlit run app.py "$@"
