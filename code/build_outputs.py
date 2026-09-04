"""
Build paper.tex, paper.pdf, and a self-contained paper.html from PAPER.md.

One source of truth (PAPER.md) so the three formats cannot drift apart.
Figures are referenced by filename in the Markdown as normal image links and
are resolved here: \\includegraphics for LaTeX, base64 data: URIs for HTML so
the page survives being moved or emailed.
"""
from __future__ import annotations

import base64
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

LATEX_PREAMBLE = r"""
\usepackage[margin=1.1in]{geometry}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{microtype}
\usepackage[table]{xcolor}
\usepackage[colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!50!black,
            citecolor=blue!50!black]{hyperref}
\theoremstyle{plain}
\newtheorem{theorem}{Theorem}
\newtheorem{corollary}[theorem]{Corollary}
\theoremstyle{definition}
\newtheorem{assumption}{Assumption}
\setlength{\emergencystretch}{3em}
\providecommand{\tightlist}{%
  \setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
"""

HTML_CSS = r"""
:root {
  color-scheme: light dark;
  --bg: #ffffff; --fg: #1a1a1a; --muted: #5c5c5c; --rule: #e0e0e0;
  --accent: #1f5fa9; --code-bg: #f5f5f5; --table-stripe: #fafafa;
  --figbg: #ffffff;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14161a; --fg: #e6e6e6; --muted: #a0a4ab; --rule: #2c3038;
    --accent: #7fb2f0; --code-bg: #1c1f25; --table-stripe: #191c21;
    --figbg: #f4f4f4;
  }
}
* { box-sizing: border-box; }
body {
  background: var(--bg); color: var(--fg); margin: 0;
  font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
  font-size: 17px; line-height: 1.62;
}
main { max-width: 46rem; margin: 0 auto; padding: 3.5rem 1.25rem 6rem; }
h1 { font-size: 2.1rem; line-height: 1.2; margin: 0 0 .3rem; letter-spacing: -.01em; }
h2 { font-size: 1.45rem; margin: 2.8rem 0 .8rem; padding-bottom: .3rem;
     border-bottom: 1px solid var(--rule); }
h3 { font-size: 1.13rem; margin: 2rem 0 .5rem; }
h4 { font-size: 1rem; margin: 1.4rem 0 .4rem; color: var(--muted); }
p { margin: 0 0 1.05rem; }
a { color: var(--accent); }
blockquote {
  margin: 1.3rem 0; padding: .7rem 1.1rem; border-left: 3px solid var(--accent);
  background: var(--code-bg); border-radius: 0 4px 4px 0;
}
blockquote p:last-child { margin-bottom: 0; }
code, pre {
  font-family: "SF Mono", ui-monospace, Menlo, Consolas, monospace;
  font-size: .87em;
}
code { background: var(--code-bg); padding: .12em .35em; border-radius: 3px; }
pre { background: var(--code-bg); padding: .9rem 1.1rem; border-radius: 6px;
      overflow-x: auto; line-height: 1.45; }
pre code { background: none; padding: 0; }
figure { margin: 2rem 0; text-align: center; }
figure img {
  max-width: 100%; height: auto; border-radius: 6px;
  background: var(--figbg); padding: 6px;
}
figcaption { color: var(--muted); font-size: .84rem; margin-top: .5rem;
             font-style: italic; }
.table-wrap { overflow-x: auto; margin: 1.5rem 0; }
table { border-collapse: collapse; width: 100%; font-size: .87rem;
        font-family: system-ui, -apple-system, sans-serif; }
th, td { padding: .45rem .6rem; text-align: left; border-bottom: 1px solid var(--rule); }
thead th { border-bottom: 2px solid var(--rule); font-weight: 600; white-space: nowrap; }
tbody tr:nth-child(even) { background: var(--table-stripe); }
hr { border: none; border-top: 1px solid var(--rule); margin: 2.5rem 0; }
strong { font-weight: 650; }
"""


def read_md() -> str:
    return (HERE / "PAPER.md").read_text()


def build_tex(md: str, out: Path = HERE / "paper.tex") -> Path:
    src = HERE / ".paper_tex_src.md"
    src.write_text(md)
    cmd = [
        "pandoc", str(src), "-f", "gfm+tex_math_dollars", "-t", "latex",
        "-s", "--toc", "--toc-depth=2",
        "--pdf-engine=xelatex",
        "-V", "documentclass=article", "-V", "fontsize=11pt",
        "-V", "mainfont=Palatino", "-V", "monofont=Menlo",
        "-V", "linkcolor=blue", "-H", str(_preamble_file()),
        "-o", str(out),
    ]
    subprocess.run(cmd, check=True, cwd=HERE)
    src.unlink(missing_ok=True)
    return out


def _preamble_file() -> Path:
    p = HERE / ".preamble.tex"
    p.write_text(LATEX_PREAMBLE)
    return p


def build_pdf(tex: Path = HERE / "paper.tex") -> Path | None:
    # xelatex, not pdflatex: the prose uses real Unicode (minus signs, arrows,
    # Greek) that pdflatex's 8-bit engine rejects outright.
    for _ in range(2):   # twice, so the ToC resolves
        r = subprocess.run(
            ["xelatex", "-interaction=nonstopmode", tex.name],
            cwd=HERE, capture_output=True, text=True)
    pdf = HERE / "paper.pdf"
    if pdf.exists():
        for ext in (".aux", ".log", ".out", ".toc"):
            (HERE / f"paper{ext}").unlink(missing_ok=True)
        return pdf
    print("pdflatex failed; tail of log:")
    print((r.stdout or "")[-1800:])
    return None


def _embed_images(html: str) -> str:
    """Replace <img src="figures/x.png"> with a base64 data: URI so the page
    is genuinely self-contained."""
    def repl(m):
        src = m.group(1)
        path = (HERE / src).resolve()
        if not path.is_file():
            return m.group(0)
        b64 = base64.b64encode(path.read_bytes()).decode()
        return m.group(0).replace(src, f"data:image/png;base64,{b64}")
    return re.sub(r'<img[^>]*src="([^"]+)"', repl, html)


def build_html(md: str, out: Path = HERE / "paper.html") -> Path:
    src = HERE / ".paper_html_src.md"
    src.write_text(md)
    css = HERE / ".paper.css"
    css.write_text(HTML_CSS)
    body = subprocess.run(
        ["pandoc", str(src), "-f", "gfm+tex_math_dollars", "-t", "html5",
         "--mathml", "--section-divs"],
        check=True, capture_output=True, text=True, cwd=HERE).stdout
    body = re.sub(r"<table>", '<div class="table-wrap"><table>', body)
    body = re.sub(r"</table>", "</table></div>", body)
    body = _embed_images(body)
    title = "Infinite-Horizon Diversity"
    page = (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>{title}</title>\n<style>{HTML_CSS}</style>\n</head>\n"
        f"<body>\n<main>\n{body}\n</main>\n</body>\n</html>\n"
    )
    out.write_text(page)
    src.unlink(missing_ok=True)
    css.unlink(missing_ok=True)
    return out


def main():
    md = read_md()
    want = sys.argv[1:] or ["tex", "pdf", "html"]
    if "tex" in want or "pdf" in want:
        tex = build_tex(md)
        print("wrote", tex.name, f"({tex.stat().st_size // 1024} KB)")
    if "pdf" in want:
        pdf = build_pdf()
        if pdf:
            print("wrote", pdf.name, f"({pdf.stat().st_size // 1024} KB)")
    if "html" in want:
        html = build_html(md)
        print("wrote", html.name, f"({html.stat().st_size // 1024} KB)")
    (HERE / ".preamble.tex").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
