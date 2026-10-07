"""Anonymized code and data release for MICCAI double-blind review.

    python paper/miccai/make_anon_release.py --out <empty folder outside this repo>

Copies what the paper says is released (the per-tooth benchmark and
scorer) plus the uncropped cross-validation code, frozen rules, folds,
boxes and result CSVs, keeping their paths so internal references still
work. RESULTS.md is cut to Sections 60 onward (the ones the paper uses).
Example X-ray PNGs are left out. Identifying strings are replaced, and the
script fails if any listed identifier is left in any copied file.
"""
import argparse
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIRST_SECTION = 60
REPLACE = [
    (r'USER = "christopherhuang88"', 'USER = "<kaggle-username>"'),
    (r'REPO_URL = "https://github\.com/christopherh-88/tooth-numbering-priors\.git"',
     'REPO_URL = "<repository-url>"'),
    (r"\s*Whether the CJSJ revision uses it is the user's call\.", ""),
    (r"\s*\(thresholds set on measured anchors at\s+the user's request\)", " (thresholds set on measured anchors)"),
    (r"\bThe user\b", "The authors"),
    (r"\bthe user\b", "the authors"),
]
FORBIDDEN = re.compile(r"christopher|huang|christopherh-88|cjsj|columbia|tooth-numbering-priors|/Users/|"
                       r"claude|anthropic|the user\b|user's", re.I)
README = """# Position-only baseline for tooth numbering: code and data

Anonymized release accompanying a MICCAI submission.

- `benchmark/`: the per-tooth table (`teeth.csv`), folds and scorer for
  comparing a tooth-numbering model with the position-only baseline. Start
  with `benchmark/README.md`.
- `cpu_repro/cv/`: the uncropped five-fold cross-validation code. Each
  experiment's decision rule is in a `*_RULES.md` file, frozen before it
  ran; `cpu_repro/cv/README.md` maps scripts to rules and result sections.
  Result CSVs are in `cpu_repro/cv/results/`.
- `RESULTS.md`: the result log for these experiments (Sections 60 onward;
  references to lower section numbers point to an earlier log on
  randomly cropped splits, not included).

The X-rays are not included. They are the UFBA-425 dataset (Budagam et
al., 2025, doi:10.6084/m9.figshare.29827475). Training and image
interventions ran as Kaggle GPU kernels written by
`cpu_repro/cv/kaggle/make_kernels.py` and `make_phase3_kernels.py`; set
`USER` and `REPO_URL` in those files before use.

Licensed under Apache 2.0 (`LICENSE`), except the files derived from the
DENTEX labels (Hamamci et al., 2023; every file with `dentex` in its path,
for example `cpu_repro/cv/dentex_boxes.csv` and
`cpu_repro/cv/results/kaggle_dentex*/`), which remain under DENTEX's
CC BY-NC-SA 4.0 license: attribution, non-commercial use only, and
derivatives under the same license. The DENTEX X-rays are not included;
they are available from the DENTEX challenge.
"""


def tracked(*paths):
    out = subprocess.run(["git", "ls-files", *paths], cwd=ROOT, capture_output=True, text=True, check=True)
    return [p for p in out.stdout.split("\n") if p]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    out = a.out.resolve()
    assert ROOT not in out.parents and out != ROOT, "write the release outside this repository"
    assert not out.exists() or not any(out.iterdir()), f"{out} is not empty"

    files = [p for p in tracked("benchmark", "cpu_repro/cv") if not p.endswith(".png")] + ["LICENSE"]
    for p in files:
        dst = out / p
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / p, dst)

    res = (ROOT / "RESULTS.md").read_text()
    start = re.search(rf"^## {FIRST_SECTION}\. ", res, re.M).start()
    end = res.index("## Adding a new entry")
    (out / "RESULTS.md").write_text("# Results log\n\n" + res[start:end].rstrip() + "\n")
    (out / "README.md").write_text(README)

    changed = 0
    for f in out.rglob("*"):
        if not f.is_file() or f.suffix not in {".py", ".md", ".csv", ".txt", ""}:
            continue
        t = f.read_text()
        new = t
        for pat, rep in REPLACE:
            new = re.sub(pat, rep, new)
        if new != t:
            f.write_text(new)
            changed += 1
    left = []
    for f in out.rglob("*"):
        if f.is_file():
            for i, line in enumerate(f.read_bytes().decode("utf-8", "replace").splitlines(), 1):
                if FORBIDDEN.search(line):
                    left.append(f"{f.relative_to(out)}:{i}: {line.strip()[:100]}")
    n = sum(1 for f in out.rglob("*") if f.is_file())
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"{n} files, {size:.1f} MB, {changed} scrubbed")
    assert not left, "identifiers left:\n" + "\n".join(left)
    print("identifier scan: clean")


if __name__ == "__main__":
    main()
