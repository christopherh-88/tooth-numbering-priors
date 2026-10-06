"""Write a Kaggle CPU kernel that compiles the LNCS draft.

    python paper/miccai/latex/make_compile_kernel.py --out <kernels dir>
    kaggle kernels push -p <kernels dir>/tooth-numbering-miccai-latex

The kernel installs TeX Live from apt, fetches the LNCS class from CTAN,
runs pdflatex and bibtex, and writes main.pdf, main.log and pages.txt to
/kaggle/working. The source files are embedded in the script (the figure
as base64), so nothing has to be committed first.
"""
import argparse
import base64
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
USER = "christopherhuang88"
NAME = "tooth-numbering-miccai-latex"
TEXT = ["main.tex", "refs.bib"]
BINARY = ["fig_closed_gap.pdf"]

BODY = '''import base64, shutil, subprocess, zipfile, urllib.request
from pathlib import Path

W = Path("/kaggle/working/build")
W.mkdir(parents=True, exist_ok=True)
for name, text in TEXT.items():
    (W / name).write_text(text)
for name, b64 in BINARY.items():
    (W / name).write_bytes(base64.b64decode(b64))

def sh(cmd, check=True):
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=W, check=check)

subprocess.run(["apt-get", "update", "-qq"], check=True)
subprocess.run(["apt-get", "install", "-y", "-qq", "--no-install-recommends", "texlive-latex-base",
                "texlive-latex-recommended", "texlive-latex-extra", "texlive-fonts-recommended",
                "poppler-utils"], check=True)
urllib.request.urlretrieve("https://mirrors.ctan.org/macros/latex/contrib/llncs.zip", "/tmp/llncs.zip")
with zipfile.ZipFile("/tmp/llncs.zip") as z:
    for m in z.namelist():
        if m.endswith(("llncs.cls", "splncs04.bst")):
            (W / Path(m).name).write_bytes(z.read(m))
sh(["pdflatex", "-interaction=nonstopmode", "main.tex"], check=False)
sh(["bibtex", "main"], check=False)
sh(["pdflatex", "-interaction=nonstopmode", "main.tex"], check=False)
sh(["pdflatex", "-interaction=nonstopmode", "main.tex"], check=False)
out = Path("/kaggle/working")
for f in ("main.pdf", "main.log", "main.blg"):
    if (W / f).exists():
        shutil.copy(W / f, out / f)
info = subprocess.run(["pdfinfo", str(out / "main.pdf")], capture_output=True, text=True).stdout
pages = [l for l in info.splitlines() if l.startswith("Pages")]
(out / "pages.txt").write_text("\\n".join(pages) + "\\n")
print(pages, flush=True)
log = (W / "main.log").read_text(errors="replace")
for l in log.splitlines():
    if l.startswith("!") or "Warning" in l and ("undefined" in l or "Overfull" in l):
        print(l, flush=True)
print("overfull boxes:", log.count("Overfull"), flush=True)
shutil.rmtree(W)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    d = a.out / NAME
    d.mkdir(parents=True, exist_ok=True)
    text = {f: (HERE / f).read_text() for f in TEXT}
    binary = {f: base64.b64encode((HERE / f).read_bytes()).decode() for f in BINARY}
    (d / f"{NAME}.py").write_text(f"TEXT = {text!r}\nBINARY = {binary!r}\n" + BODY)
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{USER}/{NAME}", "title": NAME, "code_file": f"{NAME}.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False,
        "enable_internet": True, "dataset_sources": [], "kernel_sources": [], "competition_sources": [],
        "model_sources": []}, indent=2))
    print("wrote", d)


if __name__ == "__main__":
    main()
