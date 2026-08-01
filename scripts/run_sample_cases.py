"""Generate reproducible sample-run evidence for the AI-assisted recommendation workflow.

Runs the actual CLI workflow (src.main.run) against three valid demonstration
requests and two guardrail cases, and writes the real captured output to a
file for submission evidence. Nothing here is hand-written example output.

Usage:
    python3 scripts/run_sample_cases.py --output evidence/sample_runs.txt
"""

from __future__ import annotations

import argparse
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.main import run  # noqa: E402

DEMONSTRATION_CASES: List[Tuple[str, str]] = [
    ("Valid request 1 (workout)", "I need energetic music for a workout."),
    ("Valid request 2 (study)", "Give me calm music while I study."),
    ("Valid request 3 (mood boost)", "I feel down and want something uplifting."),
    ("Guardrail: empty/whitespace request", "   "),
    ("Guardrail: out-of-scope request", "What is the weather tomorrow?"),
]


def run_case(label: str, request_text: str) -> str:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        exit_code = run(["--request", request_text])
    output = buffer.getvalue()
    return f"=== {label} ===\nCommand: python3 -m src.main --request {request_text!r}\nExit code: {exit_code}\n\n{output}\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate sample-run evidence from the real CLI workflow.")
    parser.add_argument("--output", type=str, default=None, help="Path to write the captured output to.")
    args = parser.parse_args(argv)

    sections = [run_case(label, text) for label, text in DEMONSTRATION_CASES]
    full_report = "\n".join(sections)

    print(full_report)

    if args.output:
        output_path = REPO_ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(full_report, encoding="utf-8")
        print(f"\nSample run evidence written to: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
