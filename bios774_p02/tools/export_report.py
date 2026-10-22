"""export_report.py — tidy a report's Markdown layout and export it to PDF.

  python tools/export_report.py reports/generated_report_1.md

Layout is handled here, in code, so the agent can concentrate on content:
  * every figure gets a width chosen from its aspect ratio, so figures stay
    next to their text instead of jumping pages and leaving white gaps;
  * table columns are aligned: text left, numbers right;
  * table cells that start with "+", "-", "*" or "1." are escaped (otherwise
    they render as list items);
  * a blank line is inserted before every list and table (otherwise the
    items run together into one paragraph).
The tidy step is idempotent and rewrites the .md in place; then the PDF is
written next to it with pandoc (+ typst, or a LaTeX engine as fallback).
Exit code 0 even when no PDF converter is found — the .md is the deliverable.
"""
import argparse
import glob
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

TEXT_WIDTH_IN = 6.5          # usable text width of a letter/A4 page in inches (approx.)
MAX_FIG_H_IN = 2.6           # keep ordinary figures short enough to fit beside text
MAX_GRID_H_IN = 4.6          # image grids need more room to be legible

FENCE = re.compile(r"^\s*(```|~~~)")
LIST_ITEM = re.compile(r"^\s{0,3}([-*+]|\d+[.)])\s+\S")
TABLE_ROW = re.compile(r"^\s*\|")
SEP_ROW = re.compile(r"^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)+\|?\s*$")
IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)(\{[^}]*\})?")
NUMERIC = re.compile(r"^[\s−\-+~≈<>≤≥]*[\d.,]+\s*(%|s|min|×|x)?\s*$|^\[[^\]]*\]$|^(n/?a|—|-)$", re.I)


def png_size(path):
    try:
        with open(path, "rb") as f:
            head = f.read(24)
        if head[:8] == b"\x89PNG\r\n\x1a\n":
            w, h = struct.unpack(">II", head[16:24])
            return w, h
    except OSError:
        pass
    return None


def figure_width(img_path, report_dir):
    size = png_size((report_dir / img_path).resolve())
    if not size:
        return 70
    w, h = size
    aspect = w / max(h, 1)
    max_h = MAX_GRID_H_IN if "grid" in Path(img_path).name else MAX_FIG_H_IN
    width_in = min(TEXT_WIDTH_IN * 0.92, max_h * aspect)
    return int(round(100 * width_in / TEXT_WIDTH_IN / 5) * 5)


def split_cells(row):
    r = row.strip()
    if r.startswith("|"):
        r = r[1:]
    if r.endswith("|"):
        r = r[:-1]
    return [c.strip() for c in re.split(r"(?<!\\)\|", r)]


def escape_cell(c):
    if re.match(r"^[+*-]\s", c):
        return "\\" + c
    m = re.match(r"^(\d+)\.\s", c)
    if m:
        return f"{m.group(1)}\\." + c[len(m.group(1)) + 1:]
    return c


def tidy(md, report_dir):
    lines = md.split("\n")
    out, in_fence, i = [], False, 0
    while i < len(lines):
        line = lines[i]
        if FENCE.match(line):
            in_fence = not in_fence
            out.append(line)
            i += 1
            continue
        if in_fence:
            out.append(line)
            i += 1
            continue

        # ---- tables: header row followed by a separator row
        if TABLE_ROW.match(line) and i + 1 < len(lines) and SEP_ROW.match(lines[i + 1]):
            j = i + 2
            while j < len(lines) and TABLE_ROW.match(lines[j]):
                j += 1
            header, sep, body = lines[i], lines[i + 1], lines[i + 2:j]
            seps = split_cells(sep)
            rows = [split_cells(r) for r in body]
            new_sep = []
            for k, s in enumerate(seps):
                dashes = max(3, s.count("-"))
                col = [r[k] for r in rows if k < len(r) and r[k]]
                numeric = k > 0 and col and all(NUMERIC.match(c) for c in col)
                new_sep.append("-" * dashes + ":" if numeric else ":" + "-" * dashes)
            if out and out[-1].strip():
                out.append("")
            out.append("| " + " | ".join(escape_cell(c) for c in split_cells(header)) + " |")
            out.append("|" + "|".join(new_sep) + "|")
            for r in rows:
                out.append("| " + " | ".join(escape_cell(c) for c in r) + " |")
            if j < len(lines) and lines[j].strip():
                out.append("")
            i = j
            continue

        # ---- lists: make sure a list starts after a blank line
        if LIST_ITEM.match(line) and out and out[-1].strip() and not (
                LIST_ITEM.match(out[-1]) or out[-1].startswith((" ", "\t"))):
            out.append("")

        # ---- figures: set a width that keeps them on the page
        def _img(m):
            attrs = m.group(3) or ""
            if "width" in attrs:
                return m.group(0)
            w = figure_width(m.group(2), report_dir)
            extra = attrs[1:-1].strip()
            return f"![{m.group(1)}]({m.group(2)}){{width={w}%{(' ' + extra) if extra else ''}}}"
        line = IMAGE.sub(_img, line)
        out.append(line)
        i += 1
    return "\n".join(out)


def find_tool(name):
    hit = shutil.which(name)
    if hit:
        return hit
    patterns = [
        f"/Applications/RStudio.app/Contents/Resources/app/quarto/bin/tools/*/{name}",
        f"/Applications/RStudio.app/Contents/Resources/app/quarto/bin/tools/{name}",
        f"/Applications/quarto/bin/tools/*/{name}",
        f"/Applications/quarto/bin/tools/{name}",
        f"/opt/quarto/bin/tools/*/{name}",
        f"/usr/lib/rstudio/resources/app/quarto/bin/tools/*/{name}",
    ]
    for p in patterns:
        hits = sorted(glob.glob(p))
        if hits:
            return hits[0]
    return None


def export_pdf(md_path):
    pandoc = find_tool("pandoc")
    if not pandoc:
        return False, "pandoc not found (PATH, RStudio or Quarto bundles)"
    typst = shutil.which("typst") or (str(Path(pandoc).parent / "typst")
                                      if (Path(pandoc).parent / "typst").exists() else None) or find_tool("typst")
    pdf = md_path.with_suffix(".pdf")
    base = [pandoc, md_path.name, "-o", pdf.name, "--resource-path=."]
    attempts = []
    if typst:
        # pandoc copies images to a temp dir with absolute paths, so typst's root must be "/"
        attempts.append(base + [f"--pdf-engine={typst}", "--pdf-engine-opt=--root=/",
                                "-V", "fontsize=10.5pt",
                                # no stretched word spacing inside narrow table cells
                                "-V", "header-includes=#show table: set par(justify: false)"])
    for eng in ("xelatex", "pdflatex"):
        if shutil.which(eng):
            attempts.append(base + [f"--pdf-engine={eng}", "-V", "geometry:margin=1in", "-V", "fontsize=10pt"])
    if not attempts:
        return False, "no PDF engine found (typst, xelatex or pdflatex)"
    err = ""
    for cmd in attempts:
        r = subprocess.run(cmd, cwd=md_path.parent, capture_output=True, text=True)
        if r.returncode == 0 and pdf.exists():
            return True, f"{pdf} (engine: {Path(cmd[5].split('=', 1)[1]).name})"
        err = (r.stderr or r.stdout)[-600:]
    return False, "PDF export failed: " + err


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("report", help="path to the report .md")
    p.add_argument("--no-pdf", action="store_true", help="tidy the Markdown only")
    a = p.parse_args()
    md_path = Path(a.report).resolve()
    text = md_path.read_text()
    tidied = tidy(text, md_path.parent)
    if tidied != text:
        md_path.write_text(tidied)
    print(f"[export] tidied layout of {a.report}")
    if a.no_pdf:
        return
    ok, msg = export_pdf(md_path)
    print(f"[export] {'PDF written: ' if ok else 'no PDF — '}{msg}")
    sys.exit(0)


if __name__ == "__main__":
    main()
