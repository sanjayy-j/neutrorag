"""Capture a GENUINE timestamped transcript of the real demo sequence.

THIS IS NOT THE REQUIRED DEMO VIDEO. It is a verifiable record of actual
command execution -- every line of output below was produced by running the
command shown, on this machine, at the timestamp shown. No output is
reconstructed, edited or simulated.

Why no video: this environment has no screen recorder (ffmpeg, asciinema,
script, termtosvg all absent), no capture library beyond PIL.ImageGrab, and no
audio. More decisively, commands here run in a non-interactive shell with no
visible window, so a screen capture would record the desktop while execution
happened invisibly -- which would be a fabricated demo, not a recording of one.

    python submission/demo/capture_demo_session.py

Writes submission/demo/DEMO_TRANSCRIPT.md.
"""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "submission" / "demo" / "DEMO_TRANSCRIPT.md"

# The four beats of DEMO_SCRIPT.md, in order. Only fast, committed-artefact
# commands: the expensive runs are already recorded in results/.
BEATS = [
    ("1. The construction and the pre-registered prediction",
     "Shows the registered question and its stop condition, written before the "
     "data existed.",
     ["git", "log", "--format=%h %ad %s", "--date=short", "-1", "2551e65"]),
    ("1b. The pre-registration is 1000+ lines of predictions, not prose",
     "Size and the four research questions it fixes.",
     ["python", "-c",
      "import pathlib;t=pathlib.Path('PREREGISTRATION.md').read_text(encoding='utf-8');"
      "print(f'PREREGISTRATION.md: {len(t.splitlines())} lines');"
      "print('\\n'.join(l for l in t.splitlines() if l.startswith('### RQ')))"]),
    ("2. The prediction failing",
     "The five-condition comparison at matched expansion budget. The decomposed "
     "gate is c_i_gate.",
     ["python", "experiments/analyse_hotpot_gate.py"]),
    ("3. The mechanism -- r is an affine function of coverage alone",
     "The zero-variance components and the affine law, read from the generated "
     "manifest.",
     ["python", "-c",
      "import pathlib;"
      "print('\\n'.join(l[:160] for l in "
      "pathlib.Path('results/claims.md').read_text(encoding='utf-8').splitlines() "
      "if l.startswith('| D1-') or l.startswith('| D2 ')))"]),
    ("3b. The circularity, quantified",
     "eta^2 of the change in r explained by retrieval depth alone.",
     ["python", "experiments/diagnose_r_depth.py"]),
    ("4. The design rule and the one positive registered result",
     "The Addendum 6 primary: under hard retrieval r does predict answer "
     "correctness.",
     ["python", "-c",
      "import pathlib;"
      "print('\\n'.join(l[:200] for l in "
      "pathlib.Path('results/claims.md').read_text(encoding='utf-8').splitlines() "
      "if l.startswith('| A6 ') or l.startswith('| A6-paired')))"]),
    ("5. Reproducibility",
     "The test suite and the manifest that regenerates every figure.",
     ["python", "-m", "pytest", "-q"]),
]

NOISE = ("Warning: You are sending unauthenticated", "huggingface_hub",
         "warnings.warn", "To support symlinks", "symlinks", "Loading weights",
         "Batches", "developer mode", "UserWarning")


def clean(text: str) -> str:
    keep = [l for l in text.splitlines()
            if not any(n in l for n in NOISE)]
    return "\n".join(keep).strip()


def main() -> None:
    started = datetime.now()
    parts = [
        "# NeutroRAG — demo session transcript",
        "",
        f"**Captured {started.strftime('%Y-%m-%d %H:%M:%S')} local time** by "
        "`python submission/demo/capture_demo_session.py`.",
        "",
        "> ### THIS IS NOT THE REQUIRED DEMO VIDEO",
        "> It is a genuine transcript: every line of output below was produced "
        "by running the command shown above it, on this machine, at capture "
        "time. Nothing is reconstructed, edited or simulated. Progress bars, "
        "HuggingFace download warnings and symlink warnings are filtered out; "
        "no result line is altered.",
        ">",
        "> A video could not be recorded here. The environment has no screen "
        "recorder (`ffmpeg`, `asciinema`, `script`, `termtosvg` all absent) and "
        "no audio. More decisively, these commands run in a non-interactive "
        "shell with **no visible window**, so a screen capture would have "
        "recorded the desktop while execution happened invisibly — a fabricated "
        "demo rather than a recording of one. See "
        "`submission/FINAL_AUDIT_REPORT.md` for the exact remaining step.",
        "",
        "Follows the four beats of [`DEMO_SCRIPT.md`](../../DEMO_SCRIPT.md).",
        "",
        "---",
        "",
    ]
    for title, blurb, cmd in BEATS:
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {title}")
        t0 = time.time()
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        dt = time.time() - t0
        body = clean(r.stdout) or clean(r.stderr) or "(no output)"
        parts += [
            f"## {title}",
            "",
            blurb,
            "",
            f"`$ {' '.join(cmd)}`",
            "",
            f"*captured {ts}, exit code {r.returncode}, {dt:.1f}s*",
            "",
            "```",
            body,
            "```",
            "",
        ]
        if r.returncode != 0:
            parts += [f"> **Command exited {r.returncode}.** Recorded as-is "
                      "rather than retried or hidden.", ""]
    parts += [
        "---",
        "",
        f"Capture finished {datetime.now().strftime('%H:%M:%S')}; "
        f"{len(BEATS)} beats.",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts), encoding="utf-8", newline="\n")
    print(f"\nwrote {OUT}  ({len(OUT.read_text(encoding='utf-8').splitlines())} lines)")


if __name__ == "__main__":
    main()
