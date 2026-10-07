#!/bin/bash
# AI Clips - GPU setup script
# Run on a fresh vast.ai instance (PyTorch template):
#   curl -fsSL https://prateekchandrasrivastava.github.io/ai-clips/setup.sh | bash
# Safe to run again: it skips anything already done.

set -e
PY=/venv/main/bin/python
PIP=/venv/main/bin/pip

echo "=== 1/6 Folders ==="
mkdir -p /workspace/clips /workspace/models /workspace/hf /workspace/bin /workspace/scripts

echo "=== 2/6 ffmpeg ==="
if ! command -v ffmpeg >/dev/null 2>&1; then
  apt-get update -y && apt-get install -y ffmpeg
fi
ffmpeg -version | head -1

echo "=== 3/6 Python packages (faster-whisper, yt-dlp) ==="
$PIP install -q --root-user-action=ignore -U faster-whisper "yt-dlp[default]"

echo "=== 4/6 faster-whisper audio patch (needed with av 19) ==="
FW_DIR=$($PY -c "import faster_whisper, os; print(os.path.dirname(faster_whisper.__file__))")
sed -i 's/, metadata_errors="ignore"//' "$FW_DIR/audio.py"
echo "metadata_errors left: $(grep -c metadata_errors "$FW_DIR/audio.py" || true)"

echo "=== 5/6 Deno (JavaScript helper for yt-dlp) ==="
if [ ! -x /workspace/bin/deno ]; then
  curl -fsSL https://github.com/denoland/deno/releases/latest/download/deno-x86_64-unknown-linux-gnu.zip -o /tmp/deno.zip
  $PY -m zipfile -e /tmp/deno.zip /workspace/bin
  chmod +x /workspace/bin/deno
fi
/workspace/bin/deno --version | head -1

echo "=== 6/6 Whisper large-v3 model (about 3 GB the first time) ==="
HF_HOME=/workspace/hf $PY -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cuda', compute_type='float16', download_root='/workspace/models/whisper'); print('Whisper large-v3 loads on GPU')"

echo ""
echo "=== CHECK ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "yt-dlp $(/venv/main/bin/yt-dlp --version)"
if [ -f /workspace/bin/yt-cookies.txt ]; then echo "YouTube cookies: found"; else echo "YouTube cookies: MISSING - send yt-cookies.txt to /workspace/bin/"; fi
df -h /workspace | tail -1
echo "SETUP DONE"
