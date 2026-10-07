#!/usr/bin/env python3
"""
AI Clips - render ALL clips of one video in parallel, then answer for one clip.

n8n's "Render Clip" node runs once per clip. Each call passes every clip's spec,
the execution id, and which clip it wants. The FIRST call renders all clips at the
same time (3 at once by default); later calls find the results already made and
return instantly. Output is exactly what render_clip.py prints, so nothing after
this node needs to change.

Usage:
  /venv/main/bin/python /workspace/scripts/render_all.py <run id> <wanted clipId> <spec b64> [<spec b64> ...]

Env:
  AICLIPS_PARALLEL   how many clips to render at once (default 3)
"""
import base64
import fcntl
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
RENDER = os.path.join(HERE, "render_clip.py")
PARALLEL = max(1, int(os.environ.get("AICLIPS_PARALLEL", "3")))


def render_one(b64, result_path):
    try:
        p = subprocess.run([sys.executable, RENDER, b64], capture_output=True, text=True,
                           stdin=subprocess.DEVNULL)
        lines = [l for l in p.stdout.splitlines() if l.startswith(("OK|", "ERROR|"))]
        line = lines[-1] if lines else "ERROR|render produced no result: " + (p.stderr.strip()[-300:] or "unknown")
    except Exception as e:
        line = f"ERROR|{type(e).__name__}: {e}"
    tmp = result_path + ".tmp"
    with open(tmp, "w") as f:
        f.write(line + "\n")
    os.replace(tmp, result_path)


def main():
    if len(sys.argv) < 4:
        print("ERROR|usage: render_all.py <run id> <clipId> <spec b64> ...")
        sys.exit(1)
    run_id, wanted, specs_b64 = sys.argv[1], sys.argv[2], sys.argv[3:]

    specs = []
    for b in specs_b64:
        s = json.loads(base64.b64decode(b).decode("utf-8"))
        specs.append((str(s["clipId"]), b, s))

    folder = os.path.join(os.path.dirname(specs[0][2]["videoPath"]), "out", f"run-{run_id}")
    os.makedirs(folder, exist_ok=True)
    result = lambda cid: os.path.join(folder, f"{cid}.result")

    # Only one call does the rendering; any other call waits on the lock, then just reads.
    with open(os.path.join(folder, ".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        todo = [(cid, b) for cid, b, _ in specs if not os.path.isfile(result(cid))]
        if todo:
            with ThreadPoolExecutor(max_workers=min(PARALLEL, len(todo))) as pool:
                for cid, b in todo:
                    pool.submit(render_one, b, result(cid))
        fcntl.flock(lock, fcntl.LOCK_UN)

    path = result(wanted)
    if not os.path.isfile(path):
        print(f"ERROR|no result for clip {wanted}")
        sys.exit(1)
    line = open(path).read().strip()
    print(line)
    if line.startswith("ERROR|"):
        sys.exit(1)


if __name__ == "__main__":
    main()
