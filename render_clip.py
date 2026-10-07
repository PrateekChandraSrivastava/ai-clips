#!/usr/bin/env python3
"""
AI Clips - render one finished vertical Short (1080x1920) from a long video.

Usage (n8n sends the clip as base64 JSON made by the "Parse Clips" node):
  /venv/main/bin/python /workspace/scripts/render_clip.py <base64 spec>

What it does:
  1. Finds faces (OpenCV YuNet) and follows the main speaker with a smooth "camera operator" crop.
     If there is no face for most of the clip (e.g. gameplay), it uses a blurred-background layout instead.
  2. Punch-in zooms at the moments Gemini picked.
  3. Animated word-by-word captions (current word pops in yellow, keywords in green), hook title at the top.
  4. Emoji pops at Gemini's emoji moments, and a progress bar at the bottom.
  5. Jump cuts: removes dead-air pauses and "um/uh" fillers (keeps laughter and music,
     because it only cuts gaps that are actually quiet).
  6. Sound effects: a whoosh on every zoom, a pop on every emoji (generated, no licence issues).
  7. Background music (optional): a random track from /workspace/assets/music, kept quiet and
     automatically ducked under speech. No tracks in that folder = no music.
  8. Loudness normalised to -14 LUFS (YouTube's level), GPU (NVENC) encode when available.

Spec switches (all default true): "jumpcuts", "sfx", "music".

Output: /workspace/clips/<sourceId>/out/<clipId>.mp4
Prints ONE line:  OK|<clipId>|<path>|<seconds>|<MB>|<encoder>|<layout>|cut=<s>|music=<name>   or   ERROR|<message>
"""
import base64
import glob
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import urllib.request
import wave

import numpy as np
import cv2

OUT_W, OUT_H = 1080, 1920
ASSETS = os.environ.get("AICLIPS_ASSETS", "/workspace/assets")
FONT_DIR = f"{ASSETS}/fonts"
EMOJI_DIR = f"{ASSETS}/emoji"
SFX_DIR = f"{ASSETS}/sfx"
MUSIC_DIR = f"{ASSETS}/music"
YUNET = f"{ASSETS}/models/face_detection_yunet_2023mar.onnx"

# jump cuts
CUT_MIN_GAP = 0.45       # only pauses longer than this are tightened
PAD_BEFORE = 0.15        # breathing room kept before the next word
PAD_AFTER = 0.12         # ...and after the previous word
QUIET_DB = 22            # "quiet" = this many dB below the clip's loud speech level
FILLER = re.compile(r"^(u+h+m*|u+m+|e+r+m+|e+h+|h+m+|m+h*m+)$")

# audio mix
WHOOSH_GAIN = 0.32
POP_GAIN = 0.40
MUSIC_GAIN = 0.12        # before ducking; speech pushes it down further
MUSIC_EXT = ("*.mp3", "*.m4a", "*.wav", "*.ogg", "*.aac", "*.flac")

ZOOM_MAX = 1.14          # punch-in strength
ZOOM_IN, ZOOM_HOLD, ZOOM_OUT = 0.12, 1.4, 0.35   # seconds

YELLOW = "&H0000E0FF&"   # ASS colours are BGR
GREEN = "&H0055FF55&"


# ----------------------------------------------------------------- helpers
def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, stdin=subprocess.DEVNULL)


def probe(path):
    out = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=width,height,r_frame_rate",
               "-show_entries", "format=duration", "-of", "json", path]).stdout
    d = json.loads(out)
    s = d["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    return int(s["width"]), int(s["height"]), fps, float(d["format"]["duration"])


def pick_font():
    if os.path.isfile(f"{FONT_DIR}/Montserrat-Black.ttf"):
        return "Montserrat Black"
    if os.path.isfile(f"{FONT_DIR}/Anton-Regular.ttf"):
        return "Anton"
    return "DejaVu Sans"


def pick_encoder():
    try:
        run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "color=black:s=320x320:d=0.2",
             "-c:v", "h264_nvenc", "-f", "null", "-"])
        return "h264_nvenc", ["-c:v", "h264_nvenc", "-preset", "p6", "-tune", "hq", "-rc", "vbr",
                              "-cq", "19", "-b:v", "0", "-maxrate", "25M", "-bufsize", "50M",
                              "-profile:v", "high"]
    except subprocess.CalledProcessError:
        return "libx264", ["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-profile:v", "high", "-threads", "0"]


def ass_time(t):
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def clean_word(w):
    w = w.replace("{", "").replace("}", "").replace("\\", "").strip()
    w = w.strip('"“”')
    w = w.rstrip(".,;:")      # keep ? and ! — they add energy
    return w.upper()


def norm(w):
    return "".join(ch for ch in w.lower() if ch.isalnum())


def has_audio(path):
    out = run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
               "-of", "csv=p=0", path]).stdout.strip()
    return bool(out)


def media_duration(path):
    try:
        return float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", path]).stdout.strip())
    except Exception:
        return 0.0


# ----------------------------------------------------------------- jump cuts
def audio_levels(video, t0, dur):
    """Loudness (dB) of every 10 ms of the clip's audio."""
    p = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{t0:.3f}", "-i", video, "-t", f"{dur:.3f}",
                        "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                       capture_output=True, stdin=subprocess.DEVNULL)
    a = np.frombuffer(p.stdout, np.int16).astype(np.float32) / 32768.0
    n = len(a) // 160
    if n == 0:
        return None
    a = a[:n * 160].reshape(n, 160)
    return 20 * np.log10(np.sqrt((a ** 2).mean(axis=1)) + 1e-9)


def plan_cuts(words, dur, levels, fps, enabled):
    """Returns (keep intervals in clip time, words without fillers). Intervals are snapped to frames
    so picture and sound stay perfectly in sync after cutting."""
    spoken = [w for w in words if not FILLER.match(norm(w["w"]))]
    if not enabled or len(spoken) < 2:
        return [(0.0, dur)], spoken

    quiet_line = (np.percentile(levels, 90) - QUIET_DB) if levels is not None else None

    def is_quiet(a, b):
        if quiet_line is None:
            return False
        seg = levels[int(a * 100):max(int(a * 100) + 1, int(b * 100))]
        return len(seg) > 0 and (seg < quiet_line).mean() >= 0.8

    def has_filler(a, b):
        return any(FILLER.match(norm(w["w"])) and w["s"] >= a - 0.05 and w["e"] <= b + 0.05 for w in words)

    cuts = []
    first = spoken[0]["s"] - PAD_BEFORE
    if first > 0.3 and is_quiet(0, first):
        cuts.append((0.0, first))
    for a, b in zip(spoken, spoken[1:]):
        if b["s"] - a["e"] < CUT_MIN_GAP:
            continue
        g0, g1 = a["e"] + PAD_AFTER, b["s"] - PAD_BEFORE
        if g1 - g0 > 0.08 and (has_filler(a["e"], b["s"]) or is_quiet(g0, g1)):
            cuts.append((g0, g1))

    keep, cur = [], 0.0
    for c0, c1 in cuts:
        if c0 > cur:
            keep.append((cur, c0))
        cur = max(cur, c1)
    if cur < dur:
        keep.append((cur, dur))

    snapped = []
    for s, e in keep:
        s, e = round(s * fps) / fps, round(e * fps) / fps
        if e - s >= 2 / fps:
            snapped.append((s, e))
    return (snapped or [(0.0, dur)]), spoken


class TimeMap:
    """Converts a time in the original clip into a time in the cut clip."""

    def __init__(self, intervals):
        self.iv, c = [], 0.0
        for s, e in intervals:
            self.iv.append((s, e, c))
            c += e - s
        self.total = c

    def __call__(self, t):
        for s, e, c in self.iv:
            if t < s:
                return c          # fell inside a removed pause -> start of the next piece
            if t <= e:
                return c + (t - s)
        return self.total


# ----------------------------------------------------------------- sound
def _write_wav(path, samples, sr=48000):
    data = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data.tobytes())


def ensure_sfx():
    """Makes a whoosh and a pop once (synthesised, so there are no copyright issues)."""
    os.makedirs(SFX_DIR, exist_ok=True)
    whoosh, pop = f"{SFX_DIR}/whoosh.wav", f"{SFX_DIR}/pop.wav"
    sr = 48000
    if not os.path.isfile(whoosh):
        n = int(sr * 0.45)
        rng = np.random.default_rng(7)
        x = rng.standard_normal(n).astype(np.float32)
        t = np.arange(n) / n
        fc = 350 + 5500 * np.sin(np.pi * np.clip(t / 0.75, 0, 1)) ** 2      # filter sweeps up then down
        alpha = 1 - np.exp(-2 * np.pi * fc / sr)
        y, prev = np.empty(n, np.float32), 0.0
        for i in range(n):
            prev += alpha[i] * (x[i] - prev)
            y[i] = prev
        env = np.sin(np.pi * np.clip(t / 0.62, 0, 1)) ** 2 * np.where(t > 0.62, np.exp(-(t - 0.62) * 9), 1)
        y = y * env
        _write_wav(whoosh, 0.8 * y / (np.abs(y).max() + 1e-9))
    if not os.path.isfile(pop):
        n = int(sr * 0.11)
        t = np.arange(n) / sr
        freq = 950 * np.exp(-t * 18) + 280
        y = np.sin(2 * np.pi * np.cumsum(freq) / sr) * np.exp(-t * 38)
        y[:int(sr * 0.002)] *= np.linspace(0, 1, int(sr * 0.002))
        _write_wav(pop, 0.85 * y / (np.abs(y).max() + 1e-9))
    return whoosh, pop


def pick_music(clip_id, out_dur):
    files = sorted(f for ext in MUSIC_EXT for f in glob.glob(os.path.join(MUSIC_DIR, ext)))
    if not files:
        return None, 0.0
    h = int(hashlib.md5(str(clip_id).encode()).hexdigest(), 16)
    track = files[h % len(files)]
    mdur = media_duration(track)
    room = mdur - out_dur - 3
    offset = float((h // 7) % int(room)) if room > 5 else 0.0
    return track, offset


def build_filters(ass_path, intervals, src_audio_idx, whoosh, pop, music, out_dur):
    """Video: burn in captions. Audio: cut speech, add SFX + ducked music, normalise loudness."""
    fmt = "aformat=sample_rates=48000:channel_layouts=stereo"
    g = [f"[0:v]subtitles=filename='{ass_path}':fontsdir='{FONT_DIR}',format=yuv420p[vout]"]

    if src_audio_idx is not None:
        k = len(intervals)
        g.append(f"[{src_audio_idx}:a]{fmt},asplit={k}" + "".join(f"[i{j}]" for j in range(k)))
        for j, (s, e) in enumerate(intervals):
            fade_out = max(0.0, e - s - 0.008)
            g.append(f"[i{j}]atrim=start={s:.4f}:end={e:.4f},asetpts=PTS-STARTPTS,"
                     f"afade=t=in:d=0.008,afade=t=out:st={fade_out:.4f}:d=0.008[s{j}]")
        g.append("".join(f"[s{j}]" for j in range(k)) + f"concat=n={k}:v=0:a=1[speech]")
    else:
        g.append(f"anullsrc=r=48000:cl=stereo,atrim=duration={out_dur:.3f}[speech]")

    mix = ["[sp]"]
    if music:
        idx = music[0]
        fade_st = max(0.0, out_dur - 1.2)
        g.append("[speech]asplit=2[sp][sc]")
        g.append(f"[{idx}:a]{fmt},atrim=duration={out_dur:.3f},volume={MUSIC_GAIN},"
                 f"afade=t=in:d=0.4,afade=t=out:st={fade_st:.3f}:d=1.2[mus0]")
        g.append("[mus0][sc]sidechaincompress=threshold=0.015:ratio=12:attack=10:release=450[mus]")
        mix.append("[mus]")
    else:
        g.append("[speech]anull[sp]")

    for label, (idx, times, gain) in (("w", whoosh), ("p", pop)):
        if idx is None or not times:
            continue
        g.append(f"[{idx}:a]{fmt},volume={gain},asplit={len(times)}" + "".join(f"[{label}{j}]" for j in range(len(times))))
        for j, t in enumerate(times):
            ms = int(max(0.0, t) * 1000)
            g.append(f"[{label}{j}]adelay=delays={ms}:all=1[{label}d{j}]")
            mix.append(f"[{label}d{j}]")

    if len(mix) > 1:
        g.append("".join(mix) + f"amix=inputs={len(mix)}:duration=first:normalize=0:dropout_transition=0[mx]")
    else:
        g.append("[sp]anull[mx]")
    g.append("[mx]loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[aout]")
    return ";".join(g)


# ----------------------------------------------------------------- captions
def build_ass(words, t0, dur, hook, highlights, path, font, caption_margin):
    hl = {norm(h) for h in highlights if norm(h)}

    # group words into short punchy chunks (max 3 words / ~18 chars)
    chunks, cur = [], []
    for w in words:
        raw = w["w"]
        txt = clean_word(raw)
        if not txt:
            continue
        item = {"t": txt, "s": w["s"] - t0, "e": w["e"] - t0, "k": norm(raw) in hl}
        if cur and (len(cur) >= 3
                    or item["s"] - cur[-1]["e"] > 0.4
                    or sum(len(x["t"]) for x in cur) + len(txt) > 18
                    or cur[-1]["raw_end"] in ".?!,"):
            chunks.append(cur)
            cur = []
        item["raw_end"] = raw.strip()[-1:] if raw.strip() else ""
        cur.append(item)
    if cur:
        chunks.append(cur)

    events = []
    for ci, ch in enumerate(chunks):
        next_start = chunks[ci + 1][0]["s"] if ci + 1 < len(chunks) else dur
        for j, w in enumerate(ch):
            start = w["s"]
            end = ch[j + 1]["s"] if j + 1 < len(ch) else min(w["e"] + 0.3, next_start)
            if end - start < 0.05:
                end = start + 0.05
            parts = []
            for k, x in enumerate(ch):
                colour = GREEN if x["k"] else None
                if k == j:
                    colour = GREEN if x["k"] else YELLOW
                    parts.append("{\\c" + colour + "\\fscx100\\fscy100"
                                 "\\t(0,70,\\fscx118\\fscy118)\\t(70,150,\\fscx110\\fscy110)}"
                                 + x["t"] + "{\\r}")
                elif colour:
                    parts.append("{\\c" + colour + "}" + x["t"] + "{\\r}")
                else:
                    parts.append(x["t"])
            events.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Caption,,0,0,0,,{' '.join(parts)}")

    if hook:
        hook_txt = clean_word(hook).replace("\n", " ")
        events.insert(0, f"Dialogue: 1,{ass_time(0)},{ass_time(min(3.2, dur))},Hook,,0,0,0,,"
                         "{\\fad(120,220)\\fscx80\\fscy80\\t(0,160,\\fscx100\\fscy100)}" + hook_txt)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {OUT_W}
PlayResY: {OUT_H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{font},92,&H00FFFFFF,&H00FFFFFF,&H00000000,&H90000000,0,0,0,0,100,100,1,0,1,7,4,2,80,80,{caption_margin},1
Style: Hook,{font},78,&H00FFFFFF,&H00FFFFFF,&H26000000,&H00000000,0,0,0,0,100,100,1,0,3,18,0,8,90,90,250,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(events) + "\n")
    return len(chunks)


# ----------------------------------------------------------------- faces
def detect_faces(video, t0, dur, fps, sw, sh):
    det_w = 640
    det_h = int(round(sh * det_w / sw / 2) * 2)
    step = max(1, int(round(fps / 6)))           # ~6 checks per second
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{t0:.3f}", "-i", video, "-t", f"{dur:.3f}",
           "-vf", f"fps={fps:.4f},scale={det_w}:{det_h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stdin=subprocess.DEVNULL)
    det = cv2.FaceDetectorYN.create(YUNET, "", (det_w, det_h), 0.6, 0.3, 50)
    fsize = det_w * det_h * 3
    scale = sw / det_w
    min_face = sw * 0.035                         # ignore tiny faces (crowds, posters)
    samples, i = {}, 0
    while True:
        buf = p.stdout.read(fsize)
        if len(buf) < fsize:
            break
        if i % step == 0:
            img = np.frombuffer(buf, np.uint8).reshape(det_h, det_w, 3)
            _, faces = det.detect(img)
            lst = []
            if faces is not None:
                for f in faces:
                    x, y, w, h = [float(v) * scale for v in f[:4]]
                    if w < min_face:
                        continue
                    lst.append((x + w / 2, y + h / 2, w, h))
            samples[i] = lst
        i += 1
    p.wait()
    return samples, i


def plan_camera(samples, n, sw, sh, fps, crop_w):
    """Pick the main face over time and turn it into a smooth camera path."""
    keys = sorted(samples)
    cur, pts = None, {}
    for k in keys:
        faces = samples[k]
        if not faces:
            continue
        largest = max(faces, key=lambda f: f[2] * f[3])
        if cur is None:
            cur = largest
        else:
            near = min(faces, key=lambda f: abs(f[0] - cur[0]) + abs(f[1] - cur[1]))
            jumped = abs(near[0] - cur[0]) > sw * 0.12
            if jumped or largest[2] * largest[3] > 1.8 * near[2] * near[3]:
                cur = largest          # camera cut, or someone clearly bigger/closer
            else:
                cur = near
        pts[k] = cur

    face_ratio = len(pts) / max(1, len(keys))
    if face_ratio < 0.35:
        return None, face_ratio

    # per-frame target (hold last known position through gaps)
    tx = np.full(n, sw / 2.0)
    ty = np.full(n, sh / 2.0)
    snap = np.zeros(n, dtype=bool)
    pk = sorted(pts)
    last = pts[pk[0]]
    prev_x = last[0]
    nxt = 0
    for i in range(n):
        while nxt < len(pk) and pk[nxt] <= i:
            last = pts[pk[nxt]]
            if abs(last[0] - prev_x) > sw * 0.2:
                snap[i] = True                    # hard cut: jump straight there
            prev_x = last[0]
            nxt += 1
        tx[i], ty[i] = last[0], last[1]

    # smooth like a human camera operator: dead zone + easing
    alpha = 1 - (1 - 0.12) ** (30.0 / fps)
    dead = crop_w * 0.06
    cam = np.empty(n)
    c = tx[0]
    for i in range(n):
        if snap[i]:
            c = tx[i]
        d = tx[i] - c
        if abs(d) > dead:
            c += (d - math.copysign(dead, d)) * alpha
        cam[i] = c
    return (cam, ty), face_ratio


def zoom_at(t, cues):
    z = 1.0
    for c in cues:
        a = t - c
        if a < 0 or a > ZOOM_IN + ZOOM_HOLD + ZOOM_OUT:
            continue
        if a < ZOOM_IN:
            k = a / ZOOM_IN
        elif a < ZOOM_IN + ZOOM_HOLD:
            k = 1.0
        else:
            k = 1 - (a - ZOOM_IN - ZOOM_HOLD) / ZOOM_OUT
        k = k * k * (3 - 2 * k)          # smoothstep
        z = max(z, 1 + (ZOOM_MAX - 1) * k)
    return z


# ----------------------------------------------------------------- overlays
_emoji_cache = {}


EMOJI_FONTS = ["/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
               "/usr/share/fonts/noto/NotoColorEmoji.ttf"]


def _valid_rgba(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED) if os.path.isfile(path) else None
    if img is None or img.ndim != 3 or img.shape[2] != 4:
        return None
    return img


def _render_emoji_with_font(e, path):
    """Fallback: draw the emoji with the Noto Color Emoji font (installed by setup.sh)."""
    from PIL import Image, ImageDraw, ImageFont
    font_path = next((p for p in EMOJI_FONTS if os.path.isfile(p)), None)
    if not font_path:
        return
    font = ImageFont.truetype(font_path, 109)          # Noto Color Emoji only comes in size 109
    im = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((10, 10), e, font=font, embedded_color=True)
    box = im.getbbox()
    if box:
        im.crop(box).save(path)


def load_emoji(e):
    if e in _emoji_cache:
        return _emoji_cache[e]
    key = "_".join(f"{ord(ch):x}" for ch in e if ord(ch) != 0xFE0F)
    path = f"{EMOJI_DIR}/{key}.png"
    img = _valid_rgba(path)
    if img is None:
        os.makedirs(EMOJI_DIR, exist_ok=True)
        try:   # crisp 512px Noto emoji from Google
            urllib.request.urlretrieve(f"https://fonts.gstatic.com/s/e/notoemoji/latest/{key}/512.png", path)
        except Exception:
            pass
        img = _valid_rgba(path)
    if img is None:
        try:
            _render_emoji_with_font(e, path)
        except Exception:
            pass
        img = _valid_rgba(path)
    if img is not None:   # make it square so it scales evenly
        h, w = img.shape[:2]
        s = max(h, w)
        sq = np.zeros((s, s, 4), np.uint8)
        sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = img
        img = sq
    _emoji_cache[e] = img
    return img


def paste_rgba(dst, src, cx, cy, opacity=1.0):
    h, w = src.shape[:2]
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    x1, y1 = max(0, x0), max(0, y0)
    x2, y2 = min(dst.shape[1], x0 + w), min(dst.shape[0], y0 + h)
    if x2 <= x1 or y2 <= y1:
        return
    s = src[y1 - y0:y2 - y0, x1 - x0:x2 - x0]
    a = (s[:, :, 3:4].astype(np.float32) / 255.0) * opacity
    roi = dst[y1:y2, x1:x2].astype(np.float32)
    dst[y1:y2, x1:x2] = (s[:, :, :3] * a + roi * (1 - a)).astype(np.uint8)


def draw_emojis(frame, t, cues):
    for idx, (tc, img) in enumerate(cues):
        age = t - tc
        if img is None or age < 0 or age > 1.3:
            continue
        if age < 0.18:
            sc = 0.3 + 0.85 * age / 0.18
        elif age < 0.3:
            sc = 1.15 - 0.15 * (age - 0.18) / 0.12
        else:
            sc = 1.0
        op = 1.0 if age < 1.1 else max(0.0, 1 - (age - 1.1) / 0.2)
        size = max(8, int(230 * sc))
        em = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        cx = OUT_W * (0.78 if idx % 2 == 0 else 0.22)
        paste_rgba(frame, em, cx, OUT_H * 0.585, op)


# ----------------------------------------------------------------- main
def main():
    if len(sys.argv) < 2:
        print("ERROR|usage: render_clip.py <base64 spec>")
        sys.exit(1)
    spec = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))

    video = spec["videoPath"]
    folder = os.path.dirname(video)
    out_dir = os.path.join(folder, "out")
    os.makedirs(out_dir, exist_ok=True)
    clip_id = str(spec["clipId"])
    out_path = os.path.join(out_dir, f"{clip_id}.mp4")
    ass_path = os.path.join(out_dir, f"{clip_id}.ass")

    sw, sh, src_fps, src_dur = probe(video)
    fps = min(src_fps, 60.0)
    t0 = max(0.0, float(spec["start"]) - 0.1)
    t1 = min(src_dur, float(spec["end"]) + 0.4)
    dur = t1 - t0

    # words (clip time = seconds since t0)
    with open(os.path.join(folder, "transcript.json")) as f:
        tr = json.load(f)
    words = [{**w, "s": w["s"] - t0, "e": w["e"] - t0} for s in tr["segments"] for w in s["words"]
             if w["s"] >= t0 - 0.05 and w["e"] <= t1 + 0.05]

    # jump cuts: which parts of the clip to keep
    audio_ok = has_audio(video)
    levels = audio_levels(video, t0, dur) if audio_ok else None
    intervals, spoken = plan_cuts(words, dur, levels, fps, bool(spec.get("jumpcuts", True)))
    tmap = TimeMap(intervals)
    out_dur = tmap.total
    keep_ranges = [(int(round(s * fps)), int(round(e * fps))) for s, e in intervals]

    # crop window that turns the source into 9:16
    if sw / sh > 9 / 16:
        crop_w, crop_h = sh * 9 / 16, float(sh)
    else:
        crop_w, crop_h = float(sw), sw * 16 / 9

    samples, n_det = detect_faces(video, t0, dur, fps, sw, sh)
    plan, face_ratio = plan_camera(samples, max(1, n_det), sw, sh, fps, crop_w)
    layout = "track" if plan else "fit"
    if sw / sh <= 9 / 16 * 1.05:
        layout = "track" if plan else "native"

    # captions, zooms and emojis all move onto the cut timeline
    font = pick_font()
    caption_margin = 520 if layout != "fit" else 470
    cap_words = [{"w": w["w"], "s": tmap(w["s"]), "e": tmap(w["e"])} for w in spoken]
    build_ass(cap_words, 0.0, out_dur, spec.get("hook_title", ""), spec.get("highlight_words", []),
              ass_path, font, caption_margin)

    zoom_cues = sorted({round(tmap(float(z) - t0), 2) for z in spec.get("zoom_cues", []) if t0 <= float(z) <= t1})
    emoji_cues = []
    for c in spec.get("emoji_cues", []):
        try:
            tc = float(c["t"]) - t0
            if 0 <= tc <= dur:
                emoji_cues.append((tmap(tc), load_emoji(c["emoji"])))
        except Exception:
            pass

    # sound: effects + optional music
    use_sfx = bool(spec.get("sfx", True))
    whoosh_path, pop_path = ensure_sfx() if use_sfx else (None, None)
    whoosh_times = [max(0.0, z - 0.15) for z in zoom_cues] if use_sfx else []
    pop_times = [t for t, img in emoji_cues if img is not None] if use_sfx else []
    music_path, music_off = pick_music(clip_id, out_dur) if spec.get("music", True) else (None, 0.0)

    inputs, idx = [], 1
    src_idx = None
    if audio_ok:
        inputs += ["-ss", f"{t0:.3f}", "-t", f"{dur:.3f}", "-i", video]
        src_idx, idx = idx, idx + 1
    w_idx = p_idx = m_idx = None
    if whoosh_times:
        inputs += ["-i", whoosh_path]
        w_idx, idx = idx, idx + 1
    if pop_times:
        inputs += ["-i", pop_path]
        p_idx, idx = idx, idx + 1
    if music_path:
        inputs += ["-stream_loop", "-1", "-ss", f"{music_off:.2f}", "-i", music_path]
        m_idx, idx = idx, idx + 1

    graph = build_filters(ass_path, intervals, src_idx,
                          (w_idx, whoosh_times, WHOOSH_GAIN), (p_idx, pop_times, POP_GAIN),
                          (m_idx,) if music_path else None, out_dur)

    enc_name, enc_args = pick_encoder()
    dec = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{t0:.3f}", "-i", video, "-t", f"{dur:.3f}",
                            "-vf", f"fps={fps:.4f}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                           stdout=subprocess.PIPE, stdin=subprocess.DEVNULL)
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error",
                            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OUT_W}x{OUT_H}",
                            "-r", f"{fps:.4f}", "-i", "-",
                            *inputs,
                            "-filter_complex", graph,
                            "-map", "[vout]", "-map", "[aout]",
                            *enc_args,
                            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                            "-shortest", "-movflags", "+faststart", out_path],
                           stdin=subprocess.PIPE)

    fsize = sw * sh * 3
    upscale = OUT_H / crop_h
    i, j, r = 0, 0, 0          # i = source frame, j = output frame, r = current keep range
    while True:
        buf = dec.stdout.read(fsize)
        if len(buf) < fsize:
            break
        while r < len(keep_ranges) and i >= keep_ranges[r][1]:
            r += 1
        if r >= len(keep_ranges) or i < keep_ranges[r][0]:
            i += 1
            continue                      # this frame is inside a removed pause
        src = np.frombuffer(buf, np.uint8).reshape(sh, sw, 3)
        t = j / fps

        if layout == "fit":
            # blurred, darkened full-frame background + the whole 16:9 frame in the middle
            bg_scale = OUT_H / sh
            bw = int(sw * bg_scale)
            small = cv2.resize(src, (max(1, bw // 8), OUT_H // 8), interpolation=cv2.INTER_AREA)
            small = cv2.GaussianBlur(small, (0, 0), 6)
            bg = cv2.resize(small, (bw, OUT_H), interpolation=cv2.INTER_LINEAR)
            x0 = (bw - OUT_W) // 2
            frame = (bg[:, x0:x0 + OUT_W].astype(np.float32) * 0.55).astype(np.uint8)
            z = zoom_at(t, zoom_cues)
            fw = int(OUT_W * z)
            fh = int(sh * fw / sw)
            fg = cv2.resize(src, (fw, fh), interpolation=cv2.INTER_CUBIC)
            fx = (fw - OUT_W) // 2
            fg = fg[:, fx:fx + OUT_W]
            y = (OUT_H - fh) // 2 - 80
            y1, y2 = max(0, y), min(OUT_H, y + fh)
            frame[y1:y2] = fg[y1 - y:y2 - y]
        else:
            z = zoom_at(t, zoom_cues)
            cw, ch = crop_w / z, crop_h / z
            if plan:
                cam, ty = plan
                k = min(i, len(cam) - 1)
                cx, face_y = cam[k], ty[k]
            else:
                cx, face_y = sw / 2, sh / 2
            cy = sh / 2 + (face_y - sh / 2) * ((z - 1) / (ZOOM_MAX - 1) if ZOOM_MAX > 1 else 0)
            x0 = int(round(min(max(cx - cw / 2, 0), sw - cw)))
            y0 = int(round(min(max(cy - ch / 2, 0), sh - ch)))
            crop = src[y0:y0 + int(ch), x0:x0 + int(cw)]
            frame = cv2.resize(crop, (OUT_W, OUT_H), interpolation=cv2.INTER_CUBIC)
            if upscale * z > 1.3:
                blur = cv2.GaussianBlur(frame, (0, 0), 1.2)
                frame = cv2.addWeighted(frame, 1.35, blur, -0.35, 0)

        draw_emojis(frame, t, emoji_cues)
        bar_w = int(OUT_W * min(1.0, t / out_dur))
        cv2.rectangle(frame, (0, OUT_H - 12), (bar_w, OUT_H), (0, 212, 255), -1)

        enc.stdin.write(np.ascontiguousarray(frame).tobytes())
        i += 1
        j += 1

    dec.wait()
    enc.stdin.close()
    if enc.wait() != 0:
        raise RuntimeError("ffmpeg encode failed")

    mb = os.path.getsize(out_path) / 1e6
    music_name = os.path.basename(music_path) if music_path else "none"
    print(f"OK|{clip_id}|{out_path}|{out_dur:.1f}|{mb:.1f}|{enc_name}|{layout}"
          f"|cut={dur - out_dur:.1f}s|music={music_name}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR|{type(e).__name__}: {e}")
        sys.exit(1)
