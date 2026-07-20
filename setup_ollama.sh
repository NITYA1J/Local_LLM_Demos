#!/usr/bin/env bash
#
# setup_ollama.sh
# One-shot Ollama install + model pull for the Applied AI Summer Workshop
# JupyterHub environment. No sudo required.
#
# Pulls everything needed for BOTH workshop sessions in one go:
#   - the chat model used by this folder's modules, and
#   - the embedding models used by the later "Local LLM and RAG" session.
# Faculty run this once and are set for both, rather than waiting on a
# second multi-GB download mid-workshop. The RAG folder ships its own
# equivalent script, so running either one is sufficient.
#
# LINUX ONLY. This targets the JupyterHub. On macOS or Windows, don't run
# this - see Local_LLM_Guide.md for the installer-based setup instead.
#
# Usage:
#   bash setup_ollama.sh
#
# Safe to re-run - skips steps that are already done.
#
# NOTE ON STORAGE: models are currently downloaded to /tmp because the
# JupyterHub home directory quota is too small (~10GB) to hold them.
# /tmp lives on this container's writable layer, so it is NOT guaranteed
# to persist across a session/pod restart - treat this as a stopgap.
# Before workshop day, confirm with IT whether there's a persistent
# volume (e.g. a shared /mnt path) we should point MODELS_DIR at instead,
# or whether Ollama + models can be pre-baked into the Hub's base image
# so faculty don't re-download on every session.

set -uo pipefail

INSTALL_DIR="$HOME/ollama"
# Change this if IT provides a persistent volume (e.g. /mnt/shared/ollama-models).
# /tmp is used as a safe default with space, but may not survive a session/pod restart.
MODELS_DIR="${OLLAMA_MODELS:-/tmp/ollama-models}"
LOG_FILE="$INSTALL_DIR/ollama.log"
CHAT_MODEL="llama3.2:3b"
# Not used by this folder's modules, but pulled here so the later RAG session
# is ready to go. Keep in sync with "Local LLM and RAG/setup_ollama.sh".
EMBED_MODELS=("nomic-embed-text" "mxbai-embed-large")

say() { echo -e "\n==> $1"; }
fail() { echo -e "\nERROR: $1" >&2; exit 1; }

# --- 1. Detect architecture ---------------------------------------------
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64)  ASSET="ollama-linux-amd64.tar.zst" ;;
  aarch64) ASSET="ollama-linux-arm64.tar.zst" ;;
  *) fail "This script is for Linux (the JupyterHub), and got architecture '$ARCH'.
If you're on a Mac or Windows PC, don't use this script - see Local_LLM_Guide.md
for the normal installer. Otherwise, ask an instructor for help." ;;
esac

# --- 2. Download + extract Ollama binary (skip if already installed) ----
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR" || fail "Could not enter $INSTALL_DIR"

if [ -x "$INSTALL_DIR/bin/ollama" ]; then
  say "Ollama binary already installed at $INSTALL_DIR/bin/ollama - skipping download."
else
  say "Looking up latest Ollama release..."
  LATEST_TAG="$(curl -s https://api.github.com/repos/ollama/ollama/releases/latest | grep -m1 '"tag_name"' | sed -E 's/.*"([^"]+)".*/\1/')"
  [ -n "$LATEST_TAG" ] || fail "Could not determine latest Ollama version. Check your internet connection."
  say "Downloading Ollama $LATEST_TAG for $ARCH..."
  curl -L "https://github.com/ollama/ollama/releases/download/${LATEST_TAG}/${ASSET}" -o "$ASSET" \
    || fail "Download failed."

  # Sanity check: a real archive should be several MB, not a tiny error page.
  SIZE=$(stat -c%s "$ASSET" 2>/dev/null || stat -f%z "$ASSET" 2>/dev/null || echo 0)
  [ "$SIZE" -gt 1000000 ] || fail "Downloaded file looks wrong (only $SIZE bytes). The release asset name may have changed - ask an instructor."

  say "Extracting..."
  if tar --zstd -xf "$ASSET" 2>/dev/null; then
    : # success
  else
    say "System tar lacks zstd support - falling back to Python."
    pip install --user --quiet zstandard || fail "Could not install zstandard via pip."
    python3 -c "
import zstandard, tarfile
with open('$ASSET', 'rb') as f:
    dctx = zstandard.ZstdDecompressor()
    with dctx.stream_reader(f) as reader:
        with tarfile.open(fileobj=reader, mode='r|') as t:
            t.extractall()
" || fail "Python extraction failed."
  fi
  [ -x "$INSTALL_DIR/bin/ollama" ] || fail "Extraction finished but bin/ollama not found."
fi

# --- 3. Add to PATH (idempotent, three ways) -----------------------------
# JupyterHub terminals sometimes launch bash as a login shell, which only
# reads .bash_profile/.profile (not .bashrc) unless one sources the other.
# So: write the exports to .bashrc, make sure .bash_profile and .profile
# both source .bashrc, AND symlink into ~/.local/bin (which Debian/Ubuntu's
# default .profile puts on PATH out of the box). Belt and suspenders -
# covers login/non-login shells regardless of how the Hub spawns terminals.
PATH_LINE="export PATH=\$HOME/ollama/bin:\$PATH"
MODELS_LINE="export OLLAMA_MODELS=$MODELS_DIR"
SOURCE_BASHRC='[ -f "$HOME/.bashrc" ] && . "$HOME/.bashrc"'

if ! grep -q "ollama/bin" "$HOME/.bashrc" 2>/dev/null; then
  echo "$PATH_LINE" >> "$HOME/.bashrc"
fi
# Rewrite rather than skip: if someone re-runs this with a different
# MODELS_DIR (e.g. after IT provides a persistent volume), a stale export
# left in .bashrc would silently keep winning in new shells.
sed -i '/^export OLLAMA_MODELS=/d' "$HOME/.bashrc" 2>/dev/null
echo "$MODELS_LINE" >> "$HOME/.bashrc"

for RC in "$HOME/.bash_profile" "$HOME/.profile"; do
  touch "$RC"
  if ! grep -qF "$SOURCE_BASHRC" "$RC" 2>/dev/null; then
    echo "$SOURCE_BASHRC" >> "$RC"
  fi
done

mkdir -p "$HOME/.local/bin"
ln -sf "$INSTALL_DIR/bin/ollama" "$HOME/.local/bin/ollama"

export PATH="$INSTALL_DIR/bin:$PATH"

# --- 4. Set model storage location (idempotent) ---------------------------
mkdir -p "$MODELS_DIR" || fail "Could not create model directory at $MODELS_DIR"
export OLLAMA_MODELS="$MODELS_DIR"
say "Model storage: $MODELS_DIR"

# --- 5. Start the server if it isn't already running -----------------------
if curl -s -o /dev/null http://127.0.0.1:11434; then
  say "Ollama server already running."
else
  say "Starting Ollama server..."
  nohup ollama serve > "$LOG_FILE" 2>&1 &
  disown

  say "Waiting for server to come up..."
  for i in $(seq 1 30); do
    if curl -s -o /dev/null http://127.0.0.1:11434; then
      break
    fi
    sleep 1
  done
  curl -s -o /dev/null http://127.0.0.1:11434 || fail "Server did not start. Check $LOG_FILE for details."
fi

# --- 6. Pull the models needed for both workshop sessions ------------------
say "Pulling $CHAT_MODEL (chat model, ~2GB)..."
ollama pull "$CHAT_MODEL" || fail "Failed to pull $CHAT_MODEL - check disk space with 'df -h $MODELS_DIR'."

# Embedding models aren't used by this folder's modules - they're for the
# later RAG session. Pulled now so that session needs no extra download.
for EMBED_MODEL in "${EMBED_MODELS[@]}"; do
  say "Pulling $EMBED_MODEL (embedding model, for the later RAG session)..."
  ollama pull "$EMBED_MODEL" || fail "Failed to pull $EMBED_MODEL - check disk space with 'df -h $MODELS_DIR'."
done

# --- 7. Confirm -------------------------------------------------------------
say "Installed models:"
ollama list

say "Done! Ollama is running and all models are ready."
echo "If you open a new terminal later, just run: nohup ollama serve > $LOG_FILE 2>&1 & disown"
