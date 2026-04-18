#!/bin/bash
# Train OpenWakeWord model(s) on a Vast.ai GPU instance — native install (no Docker on remote).
#
# Usage:
#   ./vast_train.sh "Horus"                              # one model
#   ./vast_train.sh "Horus" "Disengage" "Standby"        # three models, one instance
#   ./vast_train.sh "Horus" --keep                       # keep instance after
#   ./vast_train.sh "Horus" --instance <id>              # reuse an existing instance
#
# Prereqs:
#   uv tool install vastai   (or pip install vastai)
#   vastai set api-key <key>
#   SSH key registered with Vast: vastai create ssh-key "$(cat ~/.ssh/id_ed25519.pub)" -y

set -euo pipefail

# --- Args ---
WAKE_WORDS=()
KEEP=false
EXISTING_INSTANCE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep) KEEP=true; shift ;;
    --instance) EXISTING_INSTANCE="$2"; KEEP=true; shift 2 ;;
    -h|--help) grep '^#' "$0" | head -20; exit 0 ;;
    *) WAKE_WORDS+=("$1"); shift ;;
  esac
done
[[ ${#WAKE_WORDS[@]} -eq 0 ]] && { echo "usage: $0 <wake-word> [<wake-word>...] [--keep] [--instance <id>]"; exit 1; }
echo "Will train models for: ${WAKE_WORDS[*]}"

# --- Preflight ---
command -v vastai >/dev/null || { echo "install vastai: uv tool install vastai"; exit 1; }
command -v rsync  >/dev/null || { echo "install rsync"; exit 1; }
vastai show user >/dev/null 2>&1 || { echo "run: vastai set api-key <key>"; exit 1; }
[[ -f ~/.ssh/id_ed25519.pub ]] || { echo "need ~/.ssh/id_ed25519.pub"; exit 1; }

SSH_PUBKEY="$(cat ~/.ssh/id_ed25519.pub)"
SSH_KEY=~/.ssh/id_ed25519

# --- Obtain instance ---
if [[ -n "$EXISTING_INSTANCE" ]]; then
  INSTANCE="$EXISTING_INSTANCE"
  echo "Reusing instance $INSTANCE"
else
  echo "Searching for offers (Turing+ GPU, 12GB+ VRAM, 50GB+ disk, CUDA 12.1+)..."
  OFFER_JSON=$(vastai search offers \
    'reliability > 0.95 disk_space >= 50 gpu_ram >= 12 cuda_vers >= 12.1 compute_cap >= 750 inet_down >= 200 rentable=true' \
    -o 'dph+' --raw)

  eval "$(echo "$OFFER_JSON" | python3 -c "
import sys, json
d = json.loads(sys.stdin.read())
if not d: sys.exit('no offers matched')
o = d[0]
print(f'OFFER_ID={o[\"id\"]}')
print(f'PRICE={o[\"dph_total\"]:.4f}')
print(f'GPU={o[\"gpu_name\"].replace(\" \", \"_\")}')
print(f'DISK={o.get(\"disk_space\", 0):.0f}')
")"
  echo "Cheapest: $GPU  \$$PRICE/hr  ${DISK}GB disk  (offer $OFFER_ID)"
  read -rp "Create instance? [y/N] " confirm
  [[ "$confirm" != "y" && "$confirm" != "Y" ]] && exit 0

  CREATE_JSON=$(vastai create instance "$OFFER_ID" \
    --image nvidia/cuda:12.1.1-devel-ubuntu22.04 \
    --disk 50 \
    --ssh \
    --onstart-cmd 'apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-venv python3-pip git git-lfs ffmpeg libsndfile1 build-essential curl ca-certificates rsync espeak-ng' \
    --raw)
  INSTANCE=$(echo "$CREATE_JSON" | python3 -c "import sys,json;print(json.load(sys.stdin)['new_contract'])" 2>/dev/null || true)
  [[ -z "$INSTANCE" ]] && { echo "ERROR: create failed. Output: $CREATE_JSON"; exit 1; }
  echo "Instance $INSTANCE starting..."

  # Attach our SSH key (new instances need this explicitly)
  vastai attach ssh "$INSTANCE" "$SSH_PUBKEY" >/dev/null 2>&1 || true
fi

cleanup() {
  if [ "$KEEP" = false ]; then
    echo "Destroying instance $INSTANCE..."
    echo y | vastai destroy instance "$INSTANCE" 2>/dev/null || true
  else
    echo "Kept alive. Reuse: $0 <word> --instance $INSTANCE"
    echo "Destroy:  echo y | vastai destroy instance $INSTANCE"
  fi
}
trap cleanup EXIT

# --- Wait for SSH ---
echo "Waiting for SSH (up to 10 min)..."
SSH_HOST=""; SSH_PORT=""
for _ in $(seq 1 60); do
  INFO=$(vastai show instance "$INSTANCE" --raw 2>/dev/null || echo '{}')
  eval "$(echo "$INFO" | python3 -c "
import sys,json
d=json.load(sys.stdin)
print(f'STATUS={d.get(\"actual_status\",\"\")}')
print(f'SSH_HOST={d.get(\"ssh_host\",\"\")}')
print(f'SSH_PORT={d.get(\"ssh_port\",\"\")}')
")"
  if [[ "$STATUS" == "running" && -n "$SSH_HOST" ]]; then break; fi
  sleep 10
done
[[ -z "$SSH_HOST" ]] && { echo "instance never came up"; exit 1; }
echo "SSH: root@$SSH_HOST -p $SSH_PORT"

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=20 -o TCPKeepAlive=yes -i $SSH_KEY"
SSH="ssh $SSH_OPTS -p $SSH_PORT root@$SSH_HOST"

# Wait for onstart to finish installing packages (check pip specifically,
# since python3 is preinstalled but python3-pip is installed via onstart)
echo "Waiting for onstart bootstrap..."
for _ in $(seq 1 120); do
  if $SSH 'python3 -m pip --version && command -v git && command -v rsync' >/dev/null 2>&1; then break; fi
  sleep 5
done
$SSH 'python3 -m pip --version' >/dev/null || { echo "bootstrap failed — pip not available"; exit 1; }

# --- Upload repo ---
echo "Uploading repo..."
rsync -az --delete \
  --exclude data --exclude .git --exclude venv --exclude my_custom_model \
  --exclude '__pycache__' --exclude openwakeword \
  -e "ssh $SSH_OPTS -p $SSH_PORT" \
  ./ "root@$SSH_HOST:/root/owwt/"

# --- Build remote bootstrap + training script ---
REMOTE_WORDS=""
for w in "${WAKE_WORDS[@]}"; do REMOTE_WORDS+=" $(printf %q "$w")"; done

echo "Running bootstrap + training on remote..."
$SSH "bash -l" <<REMOTE
set -e
cd /root/owwt

# --- One-time install (idempotent) ---
if [ ! -f .bootstrap_done ]; then
  echo "=== Installing Python deps ==="
  python3 -m pip install --upgrade pip wheel 'setuptools<81'  # <81 keeps pkg_resources
  # PyTorch cu121
  python3 -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
  # Training deps (skip pyaudio — only needed for mic recording on host)
  grep -v '^pyaudio' requirements.txt > /tmp/reqs_remote.txt
  python3 -m pip install -r /tmp/reqs_remote.txt
  # Kokoro shim deps — pin transformers + hf_hub to versions that play with datasets==2.14.6
  python3 -m pip install \
    'transformers==4.38.2' \
    'huggingface_hub==0.20.3' \
    'kokoro<0.8' \
    fastapi uvicorn soundfile
  # OpenWakeWord from source + patch
  if [ ! -d openwakeword ]; then
    git clone https://github.com/dscripka/openWakeWord openwakeword
    python3 -m pip install -e ./openwakeword
    python3 patches/skip-piper-import.py openwakeword/openwakeword/train.py
  fi
  # Embedding models
  mkdir -p openwakeword/openwakeword/resources/models
  [ -f openwakeword/openwakeword/resources/models/embedding_model.onnx ] || \
    curl -L -o openwakeword/openwakeword/resources/models/embedding_model.onnx \
      'https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx'
  [ -f openwakeword/openwakeword/resources/models/melspectrogram.onnx ] || \
    curl -L -o openwakeword/openwakeword/resources/models/melspectrogram.onnx \
      'https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx'
  touch .bootstrap_done
fi

# --- Start Kokoro shim (background, idempotent) ---
mkdir -p logs
if curl -sf http://127.0.0.1:8880/v1/audio/voices >/dev/null 2>&1; then
  echo "Kokoro already running, reusing"
else
  pkill -KILL -f "uvicorn kokoro_server" 2>/dev/null || true
  sleep 3  # let port 8880 fully release
  nohup python3 -m uvicorn kokoro_server:app --host 0.0.0.0 --port 8880 > logs/kokoro.log 2>&1 &
  echo "Kokoro server starting..."
fi
KOKORO_READY=0
for i in \$(seq 1 60); do
  if curl -sf http://127.0.0.1:8880/v1/audio/voices >/dev/null 2>&1; then
    echo "Kokoro ready"; KOKORO_READY=1; break
  fi
  sleep 3
done
if [ "\$KOKORO_READY" != "1" ]; then
  echo "Kokoro failed to start"; tail -40 logs/kokoro.log; exit 1
fi

# --- Download training data (idempotent) ---
export DATA_DIR=./data
mkdir -p "\$DATA_DIR" my_real_samples my_custom_model
./setup-data.sh

# --- Train each wake word ---
export KOKORO_URL=http://127.0.0.1:8880
for w in $REMOTE_WORDS; do
  echo "=== Training: \$w ==="
  python3 train.py --wake-word "\$w" --data-dir ./data
done

pkill -f "uvicorn kokoro_server" 2>/dev/null || true
REMOTE

# --- Download models ---
echo "Downloading models..."
mkdir -p my_custom_model
rsync -az -e "ssh $SSH_OPTS -p $SSH_PORT" \
  "root@$SSH_HOST:/root/owwt/my_custom_model/" ./my_custom_model/

echo "Done. Model(s):"
ls -la my_custom_model/
