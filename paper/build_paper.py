"""Build the paper from a single hand-written source.

`source.md` is the manuscript. Figures are referenced by filename through
`{{FIG:name.png}}` tokens and numbered here by order of first appearance, so
moving a section renumbers the figures without touching the prose. Figures live
in `paper/figures/` inside this repo; nothing outside the repo is read.

Output: paper.md, paper.tex (ICLR-style), paper.pdf.

    python3 paper/build_paper.py
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent

TITLE = "Recursive Axis Conditioning for Diverse Synthetic Data Generation"

PREAMBLE = r"""
% ICLR-style page: 5.5in text block on US letter, Times, 10pt
\usepackage[letterpaper,left=1.5in,right=1.5in,top=1in,bottom=1in]{geometry}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{microtype}
\usepackage[table]{xcolor}
\usepackage{caption}
\usepackage{titlesec}
\usepackage{titling}
\usepackage[colorlinks=true,linkcolor=blue!45!black,urlcolor=blue!45!black,
            citecolor=blue!45!black]{hyperref}
\theoremstyle{plain}
\newtheorem{theorem}{Theorem}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{corollary}[theorem]{Corollary}
\theoremstyle{definition}
\newtheorem{assumption}{Assumption}
\setlength{\emergencystretch}{3em}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\captionsetup{font=small,labelfont=bf}

\titleformat{\section}{\normalfont\large\bfseries}{\thesection}{0.7em}{}
\titleformat{\subsection}{\normalfont\normalsize\bfseries}{\thesubsection}{0.6em}{}
\titleformat{\subsubsection}{\normalfont\normalsize\bfseries\itshape}{\thesubsubsection}{0.6em}{}
\titlespacing*{\section}{0pt}{1.7ex plus .6ex minus .2ex}{1.0ex plus .2ex}
\titlespacing*{\subsection}{0pt}{1.3ex plus .5ex minus .2ex}{0.8ex plus .2ex}
\titlespacing*{\subsubsection}{0pt}{1.1ex plus .4ex minus .2ex}{0.7ex plus .2ex}

\pretitle{\begin{center}\LARGE\bfseries}
\posttitle{\par\end{center}\vskip 0.4em}
\preauthor{\begin{center}\large}
\postauthor{\par\end{center}}
\predate{\begin{center}\normalsize}
\postdate{\par\end{center}\vskip 0.8em}

\newenvironment{iclrabstract}
  {\vspace{0.4em}\begin{center}{\bfseries\scshape Abstract}\end{center}
   \begin{list}{}{\setlength{\leftmargin}{0.45in}\setlength{\rightmargin}{0.45in}}\item[]\relax}
  {\end{list}\vspace{0.6em}}

% wide tables and figures shrink to the measure instead of running off the page
\newcommand{\fitwidth}[1]{\makebox[\linewidth][c]{\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{#1}}}

% Times carries no mathematical glyphs; give those code points a real font
\usepackage{newunicodechar}
\newfontfamily\symfont{STIX Two Math}[Scale=MatchLowercase]
\newunicodechar{ℝ}{{\symfont ℝ}}
\newunicodechar{ℙ}{{\symfont ℙ}}
\newunicodechar{𝔼}{{\symfont 𝔼}}
\newunicodechar{∈}{{\symfont ∈}}
\newunicodechar{⋃}{{\symfont ⋃}}
\newunicodechar{∪}{{\symfont ∪}}
\newunicodechar{⊆}{{\symfont ⊆}}
\newunicodechar{⊂}{{\symfont ⊂}}
\newunicodechar{∝}{{\symfont ∝}}
\newunicodechar{∎}{{\symfont ∎}}
\newunicodechar{≪}{{\symfont ≪}}
\newunicodechar{∼}{{\symfont ∼}}
\newunicodechar{∅}{{\symfont ∅}}
\newunicodechar{₉}{{\symfont ₉}}
\newunicodechar{₅}{{\symfont ₅}}
\newunicodechar{⁻}{{\symfont ⁻}}

\pagestyle{plain}
\setlength{\parskip}{0.45em}
\setlength{\parindent}{0pt}
"""


def number_figures(md: str) -> str:
    """Assign figure numbers by order of appearance and resolve every token."""
    order, seen = {}, 0
    for m in re.finditer(r"!\[\{\{FIG:([^}]+)\}\}", md):
        name = m.group(1)
        if name not in order:
            seen += 1
            order[name] = seen
    missing = sorted({m.group(1) for m in re.finditer(r"\{\{FIG:([^}]+)\}\}", md)}
                     - set(order))
    if missing:
        raise SystemExit(f"figure token with no image: {missing}")
    absent = sorted(n for n in order if not (HERE / "figures" / n).is_file())
    if absent:
        raise SystemExit(f"figure file missing from paper/figures: {absent}")
    return re.sub(r"\{\{FIG:([^}]+)\}\}",
                  lambda m: f"Figure {order[m.group(1)]}", md)


SCRIPT = re.compile(r"<(sup|sub)>(.*?)</\1>", re.S)


def fix_scripts(md: str) -> str:
    """Promote HTML sub/superscripts to inline maths.

    Pandoc's gfm reader passes `<sup>` through as raw HTML, which the LaTeX
    writer then drops silently, so `n^(-1/m)` typesets as `n-1/m`. Rewriting
    them as maths before pandoc sees them is the only place this can be fixed
    without hand-writing maths in the manuscript.
    """
    def one(m):
        kind, inner = m.group(1), m.group(2)
        inner = re.sub(r"[*_]", "", inner).replace("−", "-").strip()
        inner = inner.replace(" ", r"\,").replace("#", r"\#")
        return ("$^{%s}$" if kind == "sup" else "$_{%s}$") % inner
    return SCRIPT.sub(one, md)


LT = re.compile(r"\\begin\{longtable\}\[\]\{@\{\}([lrc]+)@\{\}\}(.*?)\\end\{longtable\}", re.S)


def fit_tables(tex: str) -> str:
    """Give unsized tables wrapping columns so no row runs into the margin."""
    def one(m):
        spec, body = m.group(1), m.group(2)
        n = len(spec)
        widths = [1] * n
        for row in body.split(r"\\"):
            if "&" not in row:
                continue
            for i, cell in enumerate(row.split("&")[:n]):
                widths[i] = max(widths[i], len(cell.strip()))
        w = [v ** 0.5 for v in widths]
        tot = sum(w)
        w = [x / tot for x in w]
        col = r">{\raggedright\arraybackslash}p{\dimexpr %.4f\linewidth-2\tabcolsep\relax}"
        cols = "".join(col % x for x in w)
        size = r"\footnotesize" if (n >= 4 or max(widths) > 45) else r"\small"
        return ("{" + size + "\n" + r"\begin{longtable}[]{@{}" + cols + r"@{}}"
                + body + r"\end{longtable}}")
    return LT.sub(one, tex)


def iclr_polish(tex: Path) -> None:
    """Pandoc's numbered Abstract section becomes the ICLR abstract block."""
    t = tex.read_text()
    t = re.sub(r"\\(?:sub)*section\{Abstract\}\\label\{abstract\}",
               r"\\begin{iclrabstract}", t, count=1)
    m = re.search(r"\\begin\{iclrabstract\}", t)
    if m:
        nxt = min((i for i in (t.find(f"\\{k}section{{", m.end())
                               for k in ("", "sub", "subsub")) if i > 0), default=-1)
        if nxt > 0:
            t = t[:nxt] + "\\end{iclrabstract}\n\n" + t[nxt:]
    t = fit_tables(t)
    tex.write_text(t)


def build_pdf() -> None:
    pre = HERE / ".preamble.tex"
    pre.write_text(PREAMBLE)
    subprocess.run([
        "pandoc", str(HERE / "paper.md"), "-f", "gfm+tex_math_dollars",
        "-t", "latex", "-s", "--pdf-engine=xelatex",
        "-V", "documentclass=article", "-V", "fontsize=10pt",
        "-V", "mainfont=Times New Roman", "-V", "monofont=Menlo",
        "-M", f"title={TITLE}", "-M", "date=",
        "-H", str(pre), "-o", str(HERE / "paper.tex")], check=True, cwd=HERE)
    iclr_polish(HERE / "paper.tex")
    print("paper.tex written")
    for _ in range(2):
        subprocess.run(["xelatex", "-interaction=nonstopmode", "paper.tex"],
                       cwd=HERE, capture_output=True, text=True)
    pdf = HERE / "paper.pdf"
    if pdf.exists():
        print(f"paper.pdf written ({pdf.stat().st_size // 1024} KB)")
    pre.unlink(missing_ok=True)


def main() -> None:
    md = fix_scripts(number_figures((HERE / "source.md").read_text()))
    (HERE / "paper.md").write_text(md)
    print(f"paper.md written ({len(md.split())} words)")
    build_pdf()


if __name__ == "__main__":
    main()
