#!/usr/bin/env python3
"""
AI Clips - transcribe a video with Whisper large-v3 (word-level timestamps).

Usage:
  HF_HOME=/workspace/hf /venv/main/bin/python /workspace/scripts/transcribe.py /workspace/clips/1/source.mp4

Writes next to the video:
  audio.wav        16 kHz mono audio (what Whisper listens to)
  transcript.json  every segment + every word with start/end time (used for captions)
  transcript.txt   short "[start-end] text" lines (sent to Gemini to pick the best moments)

Prints ONE line for n8n:
  OK|<segments>|<words>|<duration_sec>|<transcript.txt path>|<time taken or 'cached'>
  ERROR|<message>
"""
import json
import os
import subprocess
import sys
import time

MODEL_DIR = "/workspace/models/whisper"


def word_count(data):
    return sum(len(s["words"]) for s in data["segments"])


def main():
    if len(sys.argv) < 2:
        print("ERROR|usage: transcribe.py <video file>")
        sys.exit(1)

    video = os.path.abspath(sys.argv[1])
    if not os.path.isfile(video):
        print(f"ERROR|video not found: {video}")
        sys.exit(1)

    folder = os.path.dirname(video)
    out_json = os.path.join(folder, "transcript.json")
    out_txt = os.path.join(folder, "transcript.txt")

    # Already done for this video? Reuse it (saves GPU time on re-runs).
    if os.path.isfile(out_json) and os.path.isfile(out_txt):
        with open(out_json) as f:
            data = json.load(f)
        print(f"OK|{len(data['segments'])}|{word_count(data)}|{data['duration']:.1f}|{out_txt}|cached")
        return

    # 1) Pull out clean 16 kHz mono audio with ffmpeg (fast, avoids audio-decoder quirks).
    audio = os.path.join(folder, "audio.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-vn", "-ac", "1", "-ar", "16000", audio],
        check=True,
    )

    # 2) Transcribe on the GPU.
    from faster_whisper import WhisperModel

    model = WhisperModel("large-v3", device="cuda", compute_type="float16", download_root=MODEL_DIR)
    t0 = time.time()
    segments, info = model.transcribe(
        audio,
        language="en",
        word_timestamps=True,
        vad_filter=True,                       # skip silence / music-only parts
        vad_parameters={"min_silence_duration_ms": 500},
        beam_size=5,
        condition_on_previous_text=False,      # stops Whisper repeating itself on long videos
    )

    segs = []
    for s in segments:
        words = [
            {"w": w.word.strip(), "s": round(w.start, 2), "e": round(w.end, 2), "p": round(w.probability, 3)}
            for w in (s.words or [])
            if w.word.strip()
        ]
        segs.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip(), "words": words})

    data = {"language": info.language, "duration": round(info.duration, 2), "segments": segs}

    with open(out_json, "w") as f:
        json.dump(data, f, ensure_ascii=False)
    with open(out_txt, "w") as f:
        for s in segs:
            f.write(f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}\n")

    print(f"OK|{len(segs)}|{word_count(data)}|{info.duration:.1f}|{out_txt}|{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR|{type(e).__name__}: {e}")
        sys.exit(1)
