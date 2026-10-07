"""Compiles a figure document (see pivotfig.sty) into a cropped PDF."""
import os
import shutil
import subprocess
import sys
import tempfile

FIGURES = os.path.dirname(os.path.abspath(__file__))


def build(tex, pdf):
    """Runs pdflatex on `tex` in its own directory, so it reads its data by relative path."""
    src, name = os.path.split(os.path.abspath(tex))
    env = dict(os.environ, TEXINPUTS=FIGURES + os.pathsep + os.environ.get("TEXINPUTS", ""),
               SOURCE_DATE_EPOCH="0")
    with tempfile.TemporaryDirectory() as tmp:
        run = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                              "-output-directory", tmp, name],
                             cwd=src, env=env, capture_output=True, text=True)
        out = os.path.join(tmp, os.path.splitext(name)[0] + ".pdf")
        if run.returncode or not os.path.getsize(out):
            sys.exit(f"pdflatex failed on {tex}:\n{run.stdout[-3000:]}")
        shutil.copyfile(out, pdf)
