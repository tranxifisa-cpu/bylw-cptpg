"""Compile the standalone manuscript with the advisor's standard LaTeX packages."""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
NAME = "DRCPT_PG_Third_Optimized"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if shutil.which("pdflatex") is None:
        raise SystemExit("Install a TeX distribution with pdflatex, algorithm and algpseudocode.")
    for pass_number in range(1, 4):
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
             "-file-line-error", "-output-directory=" + str(out),
             str(ROOT / (NAME + ".tex"))], cwd=ROOT,
            capture_output=True, text=True,
        )
        (out / f"pass_{pass_number}.txt").write_text(result.stdout + result.stderr)
        if result.returncode:
            print(result.stdout[-4500:])
            raise SystemExit(result.returncode)
    log = (out / (NAME + ".log")).read_text(errors="replace")
    forbidden = ("There were undefined references", "There were undefined citations",
                 "multiply defined", "Overfull ", "destination with the same identifier")
    problems = [text for text in forbidden if text in log]
    if problems:
        raise SystemExit("Inspect the compilation log: " + ", ".join(problems))
    pdf = out / (NAME + ".pdf")
    if not pdf.is_file() or pdf.stat().st_size < 10000:
        raise SystemExit("Compilation did not produce a complete PDF.")
    target = ROOT / pdf.name
    if target != pdf:
        shutil.copyfile(pdf, target)
    print(json.dumps({"pdf": str(target), "bytes": target.stat().st_size,
                      "passes": 3, "unresolved_references": False}))

if __name__ == "__main__":
    main()
