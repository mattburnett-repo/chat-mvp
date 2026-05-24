"""
✨ RAG Evaluation Best Practices
Evaluation Strategy
• Create diverse test sets covering edge cases
• Use both automatic and human evaluation
• Track metrics continuously in production
• Establish baseline performance metrics
Optimization Approach
• Start with quality, then optimize for cost
• Use A/B testing for significant changes
• Monitor user satisfaction alongside metrics
• Implement gradual rollouts for safety
"""

# Run numbered evaluation scripts (01-05) with the project venv
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import TextIO

# tqdm / RAGAS progress lines and ANSI codes break default pdflatex
_ANSI_ESCAPE = re.compile(r"\x1B\[[0-9;]*[a-zA-Z]")
_TQDM_PROGRESS = re.compile(r"Evaluating:\s*\d+%\|")
_BLOCK_CHARS = str.maketrans("█▏▎▍▌▋▊▉░", "########-")
# pdflatex only accepts ASCII; map common symbols, then drop anything left
_PDF_SYMBOLS = str.maketrans(
    {
        "\ufe0f": "",  # emoji variation selector (e.g. after ⚠)
        "⚠": "[!]",
        "✓": "[ok]",
        "✔": "[ok]",
        "✗": "[x]",
        "•": "-",
        "…": "...",
        "–": "-",
        "—": "-",
        "↑": "^",
        "↓": "v",
        "→": "->",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
)

LESSON_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LESSON_DIR.parent
PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
THIS_SCRIPT = Path(__file__).name
REPORT_TXT = LESSON_DIR / "run-report.txt"
REPORT_PDF = LESSON_DIR / "run-report.pdf"

# PDF uses pdflatex + Computer Modern (default TeX font; ships with MacTeX/TeX Live).
# No custom font packages — avoids DejaVu/Helvetica/fontspec issues on macOS.
PDF_ENGINE = "pdflatex"
QUICK_SKIP = {"01-RAG-evaluation-metrics.py"}


def _scripts_to_run(*, quick: bool = False) -> list[Path]:
    scripts = sorted(LESSON_DIR.glob("[0-9]*.py"))
    if quick:
        scripts = [p for p in scripts if p.name not in QUICK_SKIP]
    return scripts


def _ask_pdf_export() -> bool:
    try:
        answer = input("Output entire run to PDF? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in ("y", "yes")


def _emit(message: str, log: TextIO | None = None) -> None:
    print(message)
    if log is not None:
        log.write(message + "\n")
        log.flush()


def _ascii_for_pdf(text: str) -> str:
    text = text.translate(_PDF_SYMBOLS)
    return text.encode("ascii", "replace").decode()


def _sanitize_log_for_pdf(txt_path: Path) -> None:
    """Make log ASCII-safe for pdflatex (strip ANSI, tqdm, Unicode symbols)."""
    text = txt_path.read_text(encoding="utf-8", errors="replace")
    text = _ANSI_ESCAPE.sub("", text)
    lines: list[str] = []
    for line in text.splitlines():
        if _TQDM_PROGRESS.search(line):
            continue
        if any(ch in line for ch in "█▏▎▍▌▋▊▉░") and "it/s" in line:
            continue
        lines.append(_ascii_for_pdf(line.translate(_BLOCK_CHARS)))
    txt_path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _run_script(
    path: Path,
    log: TextIO | None = None,
    *,
    quiet_progress: bool = False,
    update_baseline: bool = False,
) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(LESSON_DIR), env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    if quiet_progress:
        env["TQDM_DISABLE"] = "1"
    if update_baseline:
        env["EVAL_UPDATE_BASELINE"] = "1"

    proc = subprocess.Popen(
        [str(PYTHON), "-W", "ignore", str(path)],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if log is not None:
            log.write(line)
            log.flush()
    return proc.wait()


def _txt_to_pdf(txt_path: Path, pdf_path: Path) -> bool:
    pandoc = shutil.which("pandoc")
    if not pandoc:
        return False
    if not shutil.which(PDF_ENGINE):
        print(
            f"Missing {PDF_ENGINE} (install MacTeX or TeX Live for PDF export).",
            file=sys.stderr,
        )
        return False

    _sanitize_log_for_pdf(txt_path)

    result = subprocess.run(
        [
            pandoc,
            str(txt_path),
            "-o",
            str(pdf_path),
            "--pdf-engine",
            PDF_ENGINE,
            "-V",
            "geometry:margin=1in",
            "-V",
            "fontsize=10pt",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stderr or result.stdout, file=sys.stderr)
        return False
    return pdf_path.is_file()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run evaluation scripts 01-05.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip expensive RAGAS script (01-RAG-evaluation-metrics.py).",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Write metric means to evaluations/lib/baseline-scores.json.",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Capture output to run-report.txt and convert to PDF.",
    )
    args = parser.parse_args()

    if not PYTHON.is_file():
        print(f"Missing venv interpreter: {PYTHON}", file=sys.stderr)
        return 1

    scripts = _scripts_to_run(quick=args.quick)
    if not scripts:
        print("No scripts to run.")
        return 0

    export_pdf = args.pdf or _ask_pdf_export()
    log_file: TextIO | None = None

    if export_pdf:
        REPORT_TXT.write_text("", encoding="utf-8")
        log_file = REPORT_TXT.open("w", encoding="utf-8")
        _emit("PDF export enabled.", log_file)
        _emit(f"Capturing output to: {REPORT_TXT}", log_file)
        _emit(f"Started: {datetime.now().isoformat(timespec='seconds')}", log_file)
        _emit("", log_file)

    _emit(f"Project: {PROJECT_ROOT}", log_file)
    _emit(f"Python:  {PYTHON}", log_file)
    _emit(f"Running {len(scripts)} script(s) in {LESSON_DIR.name}/", log_file)
    if args.quick:
        _emit("(quick mode: skipping 01-RAG-evaluation-metrics.py)", log_file)
    _emit("", log_file)

    outcomes: list[tuple[str, int]] = []

    for index, path in enumerate(scripts, start=1):
        _emit("=" * 60, log_file)
        if export_pdf:
            _emit(
                f"[{index}/{len(scripts)}] Running: {path.name}",
                log_file,
            )
        else:
            _emit(f"▶ {path.name}", log_file)
        _emit("=" * 60, log_file)

        code = _run_script(
            path,
            log_file,
            quiet_progress=export_pdf,
            update_baseline=args.update_baseline,
        )
        outcomes.append((path.name, code))
        _emit("", log_file)

    _emit("=" * 60, log_file)
    _emit("Summary", log_file)
    _emit("=" * 60, log_file)
    failed = 0
    for name, code in outcomes:
        status = "OK" if code == 0 else f"FAILED (exit {code})"
        _emit(f"  {name}: {status}", log_file)
        if code != 0:
            failed += 1

    if failed:
        _emit(f"\n{failed} script(s) failed.", log_file)
        exit_code = 1
    else:
        _emit("\nAll scripts completed successfully.", log_file)
        exit_code = 0

    if export_pdf and log_file is not None:
        _emit(f"Finished: {datetime.now().isoformat(timespec='seconds')}", log_file)
        log_file.close()
        log_file = None

        print("\nConverting run log to PDF (pdflatex / Computer Modern)...")
        if _txt_to_pdf(REPORT_TXT, REPORT_PDF):
            print(f"PDF saved: {REPORT_PDF}")
        else:
            print("Could not build PDF automatically.", file=sys.stderr)
            if not shutil.which("pandoc"):
                print(
                    "Install pandoc (https://pandoc.org/installing.html), then run:",
                    file=sys.stderr,
                )
                print(
                    f"  pandoc {REPORT_TXT} -o {REPORT_PDF}",
                    file=sys.stderr,
                )
            print(
                f"Or open {REPORT_TXT} in TextEdit -> Print -> Save as PDF.",
                file=sys.stderr,
            )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
