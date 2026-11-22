"""Regenerate every figure, table and report number from the saved results,
without rerunning any experiment:

    python plots.py

Reads results/*.json|*.pkl, writes figures/*.pdf|png, results/*_table.tex,
results/e1_sizes.tex, results/e1_fb1.tex and results/numbers.tex (the LaTeX
macros used for every result number quoted in the report text).
"""
import os, subprocess, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(ROOT, "experiments")

if __name__ == "__main__":
    rc = subprocess.call([sys.executable, "make_figures.py"] + sys.argv[1:], cwd=EXP)
    if rc == 0 and len(sys.argv) == 1:
        rc = subprocess.call([sys.executable, "report_numbers.py"], cwd=EXP)
    sys.exit(rc)
