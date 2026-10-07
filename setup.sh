#!/bin/bash
# AI Clips - GPU setup script
# Run on a fresh vast.ai instance (PyTorch template):
#   curl -fsSL "https://prateekchandrasrivastava.github.io/ai-clips/setup.sh?v=$(date +%s)" | bash
# Safe to run again: it skips anything already done and refreshes the workflow scripts.

set -e
PY=/venv/main/bin/python
PIP=/venv/main/bin/pip
A=/workspace/assets

echo "=== 1/8 Folders ==="
mkdir -p /workspace/clips /workspace/models /workspace/hf /workspace/bin /workspace/scripts \
         $A/fonts $A/models $A/emoji

echo "=== 2/8 System tools (ffmpeg, colour emoji font) ==="
NEED_APT=""
command -v ffmpeg >/dev/null 2>&1 || NEED_APT="$NEED_APT ffmpeg"
[ -f /usr/share/fonts/truetype/noto/NotoColorEmoji.ttf ] || NEED_APT="$NEED_APT fonts-noto-color-emoji"
if [ -n "$NEED_APT" ]; then
  apt-get update -y -qq </dev/null && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $NEED_APT </dev/null
fi
ffmpeg -nostdin -version </dev/null | head -1

echo "=== 3/8 Python packages ==="
$PIP install -q --root-user-action=ignore -U faster-whisper "yt-dlp[default]" pillow </dev/null
if ! $PY -c "import cv2; cv2.FaceDetectorYN" >/dev/null 2>&1; then
  $PIP install -q --root-user-action=ignore opencv-python-headless </dev/null
fi
$PY -c "import cv2; print('opencv', cv2.__version__)"

echo "=== 4/8 faster-whisper audio patch (needed with av 19) ==="
FW_DIR=$($PY -c "import faster_whisper, os; print(os.path.dirname(faster_whisper.__file__))")
sed -i 's/, metadata_errors="ignore"//' "$FW_DIR/audio.py"
echo "metadata_errors left: $(grep -c metadata_errors "$FW_DIR/audio.py" || true)"

echo "=== 5/8 Deno (JavaScript helper for yt-dlp) ==="
if [ ! -x /workspace/bin/deno ]; then
  curl -fsSL https://github.com/denoland/deno/releases/latest/download/deno-x86_64-unknown-linux-gnu.zip -o /tmp/deno.zip
  $PY -m zipfile -e /tmp/deno.zip /workspace/bin
  chmod +x /workspace/bin/deno
fi
/workspace/bin/deno --version | head -1

echo "=== 6/8 Whisper large-v3 model (about 3 GB the first time) ==="
HF_HOME=/workspace/hf $PY -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cuda', compute_type='float16', download_root='/workspace/models/whisper'); print('Whisper large-v3 loads on GPU')"

echo "=== 7/8 Editing assets (caption font, face detector) ==="
if [ ! -s $A/fonts/Montserrat-Black.ttf ]; then
  curl -fsSL https://raw.githubusercontent.com/JulietaUla/Montserrat/master/fonts/ttf/Montserrat-Black.ttf -o $A/fonts/Montserrat-Black.ttf
fi
YUNET=$A/models/face_detection_yunet_2023mar.onnx
if [ ! -f $YUNET ] || [ $(stat -c%s $YUNET) -lt 100000 ]; then
  curl -fsSL https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx -o $YUNET
fi
echo "  font: $(stat -c%s $A/fonts/Montserrat-Black.ttf) bytes, face model: $(stat -c%s $YUNET) bytes"

echo "=== 8/8 Workflow scripts (always refreshed to the latest version) ==="
BASE=https://prateekchandrasrivastava.github.io/ai-clips
for f in transcribe.py render_clip.py render_all.py; do
  curl -fsSL "$BASE/$f?v=$(date +%s)" -o "/workspace/scripts/$f"
  echo "  got $f"
done

echo ""
echo "=== CHECK ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "yt-dlp $(/venv/main/bin/yt-dlp --version)"
if ffmpeg -nostdin -v error -f lavfi -i color=black:s=320x320:d=0.2 -c:v h264_nvenc -f null - </dev/null >/dev/null 2>&1; then
  echo "GPU video encoding (NVENC): yes"
else
  echo "GPU video encoding (NVENC): no - will use CPU encoding (slower, same quality)"
fi
if [ -f /workspace/bin/yt-cookies.txt ]; then echo "YouTube cookies: found"; else echo "YouTube cookies: MISSING - send yt-cookies.txt to /workspace/bin/"; fi
df -h /workspace | tail -1
echo "SETUP DONE"
