"""Generate paper.tex from PAPER.md + figures/*.png.

Same single-source principle as build_html.py: PAPER.md is authoritative and
both output formats are rendered from it, so the three deliverables cannot
drift apart. Handles the markdown subset PAPER.md uses:

  - ATX headers  -> \\section / \\subsection / \\subsubsection
  - <!--FIG:name.png|caption--> -> figure environment with \\includegraphics
  - pipe tables  -> booktabs tabular (column count inferred, wrapped in
                    \\resizebox when wide)
  - blockquotes beginning "**Proposition N (name).**" -> proposition
                    environment (amsthm); other blockquotes -> quote
  - **bold**, *italic*, `code`, lists, rules
  - Unicode used in the prose (Greek, math operators, arrows, dashes, curly
    quotes) is mapped to LaTeX so the file compiles under pdflatex.

Build:  pdflatex paper.tex   (twice, for references)
"""
from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{microtype}
\usepackage[hidelinks]{hyperref}
\usepackage{caption}

\theoremstyle{plain}
\newtheorem{proposition}{Proposition}
\theoremstyle{definition}
\newtheorem{remark}{Remark}

\captionsetup{font=small}
\setlength{\parskip}{0.4em}
\setlength{\parindent}{0pt}

\title{%(title)s}
\author{}
\date{}

\begin{document}
\maketitle
"""

UNICODE_MAP = {
    "\u2014": "---", "\u2013": "--", "\u2018": "`", "\u2019": "'",
    "\u201c": "``", "\u201d": "''", "\u2026": "\\ldots{}",
    "\u00d7": "$\\times$", "\u2264": "$\\leq$", "\u2265": "$\\geq$",
    "\u2260": "$\\neq$", "\u2248": "$\\approx$", "\u221e": "$\\infty$",
    "\u2212": "$-$", "\u00b1": "$\\pm$", "\u2192": "$\\to$",
    "\u2229": "$\\cap$", "\u222a": "$\\cup$", "\u2282": "$\\subset$",
    "\u2286": "$\\subseteq$", "\u2208": "$\\in$", "\u2211": "$\\sum$",
    "\u221a": "$\\sqrt{\\,}$", "\u00b7": "$\\cdot$", "\u2032": "$'$",
    "\u03b5": "$\\varepsilon$", "\u03bc": "$\\mu$", "\u03b4": "$\\delta$",
    "\u03c3": "$\\sigma$", "\u03c6": "$\\varphi$", "\u03bb": "$\\lambda$",
    "\u0398": "$\\Theta$", "\u2113": "$\\ell$", "\u00b2": "$^2$",
    "\u00b3": "$^3$", "\u207b": "$^-$", "\u00b0": "$^\\circ$",
    "\u2205": "$\\emptyset$", "\u211d": "$\\mathbb{R}$", "\u00bd": "$1/2$",
    "\u2020": "\\dag{}", "\u2261": "$\\equiv$", "\u223c": "$\\sim$",
    "\u00a0": "~",
}


UNICODE_MAP.update({"\u00a7": "\\S{}", "\u22c3": "$\\bigcup$",
                    "\u2016": "$\\|$", "\u220e": "\\qedsymbol",
                    "F\u0302": "$\\hat{F}$", "\u0302": "",
                    "\u221d": "$\\propto$", "\u03a3": "$\\Sigma$",
                    "\u226a": "$\\ll$", "\u2227": "$\\wedge$",
                    "\u00b9": "$^1$"})

# Same characters, but as bare commands for use INSIDE math mode.
MATH_UNICODE = {
    "\u03b5": r"\varepsilon", "\u03bc": r"\mu", "\u03b4": r"\delta",
    "\u03c3": r"\sigma", "\u03c6": r"\varphi", "\u03bb": r"\lambda",
    "\u0398": r"\Theta", "\u2113": r"\ell", "\u211d": r"\mathbb{R}",
    "\u2264": r"\leq", "\u2265": r"\geq", "\u2260": r"\neq",
    "\u2248": r"\approx", "\u221e": r"\infty", "\u2212": "-",
    "\u00b1": r"\pm", "\u2192": r"\to", "\u00d7": r"\times",
    "\u2229": r"\cap", "\u222a": r"\cup", "\u22c3": r"\bigcup",
    "\u2282": r"\subset", "\u2286": r"\subseteq", "\u2208": r"\in",
    "\u2211": r"\sum", "\u00b7": r"\cdot", "\u2032": "'",
    "\u2205": r"\emptyset",
    "\u2261": r"\equiv", "\u223c": r"\sim", "\u2026": r"\ldots",
        "\u00bd": r"\tfrac{1}{2}", "\u221d": r"\propto", "\u03a3": r"\Sigma",
    "\u226a": r"\ll", "\u2227": r"\wedge",
}

# Any non-ASCII surviving both maps is a bug: fail loudly rather than emitting
# a .tex that pdflatex will reject deep into a long build.
def assert_ascii(tex: str) -> None:
    import collections, unicodedata
    bad = collections.Counter(ch for ch in tex if ord(ch) > 127)
    if bad:
        detail = ", ".join(f"{ch!r} (U+{ord(ch):04X} {unicodedata.name(ch, '?')}) x{n}"
                           for ch, n in bad.most_common())
        raise SystemExit(f"build_tex: unmapped non-ASCII in output: {detail}")

# Tokens that should render as math even when they appear in running prose:
# identifiers carrying a sub/superscript, and the Greek letters we use as
# variables. Matched BEFORE escaping so their braces survive.
MATH_TOKEN = re.compile(
    r"(?<![\w\\])("
    r"[A-Za-z\u03b5\u03bc\u03b4\u03c3\u03c6\u03bb]"
    r"(?:\^|_)"
    r"(?:\([^()]*\)|\{[^{}]*\}|[\w\u2212\u00b2\u00b3/+-]+)"
    r"|[A-Za-z0-9]+\u207b?[\u00b9\u00b2\u00b3\u2070\u2074-\u2079]+"
    r"|t\u2089\u2085"
    r")")


def to_math(s: str) -> str:
    """Render a fragment in math mode: map unicode to bare math commands and
    normalize the ASCII we write in prose (n^(-1/m) -> n^{-1/m})."""
    s = s.replace("F\u0302", r"\hat{F}").replace("\u2016", r"\|")
    s = s.replace("\u220e", "")
    SUB = "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089"
    SUP = "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079"
    s = re.sub("[" + SUB + "]+",
               lambda m: "_{" + "".join(str(SUB.index(c)) for c in m.group()) + "}", s)
    # a superscript RUN may open with a superscript minus (10^-3)
    s = re.sub("[\u207b" + SUP + "]+",
               lambda m: "^{" + "".join("-" if c == "\u207b" else str(SUP.index(c))
                                        for c in m.group()) + "}", s)
    # multi-symbol subscripts written with commas: F_mu,eps(S) -> F_{mu,eps}(S)
    s = re.sub(r"_([\u03b5\u03bc\u03b4\u03c3\u03bb])((?:,[\u03b5\u03bc\u03b4\u03c3\u03bb])+)",
               lambda m: "_{" + m.group(1) + m.group(2) + "}", s)
    for u, t in MATH_UNICODE.items():
        s = s.replace(u, t + " " if re.fullmatch(r"\\[a-zA-Z]{2,}", t) else t)
    s = re.sub(r"([\^_])\(([^()]*)\)", r"\1{\2}", s)
    s = re.sub(r"([\^_])([A-Za-z0-9])(?![\w{])", r"\1{\2}", s)
    return s


def esc(s: str) -> str:
    """Escape LaTeX specials, then map unicode. Code and math spans are
    protected by the caller before this runs."""
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    for u, t in UNICODE_MAP.items():
        s = s.replace(u, t)
    # straight double quotes -> TeX quotes (pairwise)
    out, open_q = [], True
    for ch in s:
        if ch == '"':
            out.append("``" if open_q else "''")
            open_q = not open_q
        else:
            out.append(ch)
    return "".join(out)


def inline(s: str) -> str:
    """Inline markdown -> LaTeX, protecting `code` and math spans."""
    spans: list[tuple[str, str]] = []

    def stash(kind, text):
        spans.append((kind, text))
        return f"\x00{len(spans) - 1}\x00"

    s = re.sub(r"`([^`]+)`", lambda m: stash("code", m.group(1)), s)
    s = MATH_TOKEN.sub(lambda m: stash("math", m.group(1)), s)
    s = esc(s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"\\emph{\1}", s)

    def unstash(m):
        kind, text = spans[int(m.group(1))]
        if kind == "code":
            return r"\texttt{" + esc(text) + "}"
        return "$" + to_math(text) + "$"

    return re.sub(r"\x00(\d+)\x00", unstash, s)


# An indented line that is really a display formula (contains a relation and
# no sentence punctuation) becomes \[ ... \].
DISPLAY_RE = re.compile(r"^\s{2,}(?=\S).*[=\u2265\u2264\u22c3].*$")


def is_display(ln: str) -> bool:
    if not DISPLAY_RE.match(ln):
        return False
    t = ln.strip().rstrip(",.")
    if len(t) > 110 or t.endswith(":"):
        return False
    words = re.findall(r"[A-Za-z]{4,}", t)
    return len(words) <= 3


def display_tex(ln: str) -> str:
    body = to_math(ln.strip().rstrip("."))
    body = body.replace("#", r"\#").replace("%", r"\%")
    body = re.sub(r"\bdist\b", r"\\mathrm{dist}", body)
    body = re.sub(r"\bvol\b", r"\\mathrm{vol}", body)
    return "\\[\n" + body + "\n\\]\n"


def figure_tex(name: str, caption: str) -> str:
    if not (FIG / name).exists():
        return f"% missing figure: {name}\n"
    return ("\\begin{figure}[htbp]\n\\centering\n"
            f"\\includegraphics[width=\\linewidth]{{figures/{name}}}\n"
            f"\\caption{{{inline(caption)}}}\n\\end{{figure}}\n")


def table_tex(rows: list[str]) -> str:
    def cells(r):
        return [c.strip() for c in r.strip().strip("|").split("|")]

    header = cells(rows[0])
    body = [cells(r) for r in rows[2:]]
    ncol = len(header)
    body = [r + [""] * (ncol - len(r)) if len(r) < ncol else r[:ncol] for r in body]
    colspec = "l" + "r" * (ncol - 1)
    wide = ncol >= 6
    out = ["\\begin{table}[htbp]\n\\centering\n\\small"]
    if wide:
        out.append("\\resizebox{\\linewidth}{!}{%")
    out.append(f"\\begin{{tabular}}{{{colspec}}}\n\\toprule")
    out.append(" & ".join(f"\\textbf{{{inline(c)}}}" for c in header) + " \\\\")
    out.append("\\midrule")
    for r in body:
        out.append(" & ".join(inline(c) for c in r) + " \\\\")
    out.append("\\bottomrule\n\\end{tabular}")
    if wide:
        out.append("}")
    out.append("\\end{table}")
    return "\n".join(out)


PROP_RE = re.compile(r"^\*\*Proposition\s+(\d+)\s*\(([^)]+)\)\.\*\*\s*(.*)$", re.S)


def quote_tex(lines: list[str]) -> str:
    text = " ".join(lines).strip()
    m = PROP_RE.match(text)
    if m:
        return ("\\begin{proposition}[" + inline(m.group(2)) + "]\n"
                + inline(m.group(3)) + "\n\\end{proposition}")
    return "\\begin{quote}\n" + inline(text) + "\n\\end{quote}"


def md_to_tex(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = []
    para: list[str] = []
    listbuf: list[str] = []
    listenv = ""
    quote: list[str] = []
    i = 0

    def flush_para():
        if para:
            out.append(inline(" ".join(para)) + "\n")
            para.clear()

    def flush_list():
        nonlocal listenv
        if listbuf:
            out.append(f"\\begin{{{listenv}}}")
            out.extend("  \\item " + inline(x) for x in listbuf)
            out.append(f"\\end{{{listenv}}}\n")
            listbuf.clear()
            listenv = ""

    def flush_quote():
        if quote:
            out.append(quote_tex(quote) + "\n")
            quote.clear()

    def flush_all():
        flush_para(); flush_list(); flush_quote()

    while i < len(lines):
        ln = lines[i]
        fig = re.match(r"\s*<!--FIG:([^|]+)\|(.*?)-->\s*$", ln)
        if fig:
            flush_all()
            out.append(figure_tex(fig.group(1).strip(), fig.group(2).strip()))
            i += 1
            continue
        if ln.startswith("|") and i + 1 < len(lines) and \
                re.match(r"^\|[\s\-|:]+\|?\s*$", lines[i + 1]):
            flush_all()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i]); i += 1
            out.append(table_tex(rows) + "\n")
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", ln)
        if m:
            flush_all()
            lvl, txt = len(m.group(1)), m.group(2)
            if lvl == 1:
                i += 1
                continue  # title handled by \maketitle
            cmd = {2: "section", 3: "subsection", 4: "subsubsection"}[lvl]
            txt = re.sub(r"^\d+(\.\d+)*\.?\s+", "", txt)  # drop manual numbering
            out.append(f"\\{cmd}{{{inline(txt)}}}")
            i += 1
            continue
        if re.match(r"^\s*---+\s*$", ln):
            flush_all()
            out.append("\\bigskip\\hrule\\bigskip\n")
            i += 1
            continue
        if is_display(ln) and not listbuf:
            flush_all()
            out.append(display_tex(ln))
            i += 1
            continue
        if ln.startswith(">"):
            flush_para(); flush_list()
            quote.append(ln.lstrip("> ").rstrip())
            i += 1
            continue
        mli = re.match(r"^[-*]\s+(.*)$", ln)
        moi = re.match(r"^\d+\.\s+(.*)$", ln)
        if mli or moi:
            flush_para(); flush_quote()
            env = "itemize" if mli else "enumerate"
            if listenv and listenv != env:
                flush_list()
            listenv = env
            listbuf.append((mli or moi).group(1))
            j = i + 1
            while j < len(lines) and lines[j].startswith("  ") and lines[j].strip() \
                    and not re.match(r"^\s*[-*]\s+|^\s*\d+\.\s+", lines[j]):
                listbuf[-1] += " " + lines[j].strip()
                j += 1
            i = j
            continue
        if not ln.strip():
            flush_all()
            i += 1
            continue
        flush_list(); flush_quote()
        para.append(ln.strip())
        i += 1
    flush_all()
    return "\n".join(out)


def main():
    md = (HERE / "PAPER.md").read_text()
    first = md.splitlines()[0]
    title = first[2:].strip() if first.startswith("# ") else "Coverage at a Finite Budget"
    tex = (PREAMBLE % {"title": inline(title)}) + md_to_tex(md) + "\n\\end{document}\n"
    assert_ascii(tex)
    (HERE / "paper.tex").write_text(tex)
    print(f"wrote paper.tex ({len(tex) / 1000:.0f} KB, "
          f"{tex.count(chr(92) + 'section')} sections, "
          f"{tex.count('begin{table}')} tables, "
          f"{tex.count('begin{figure}')} figures, "
          f"{tex.count('begin{proposition}')} propositions)")


if __name__ == "__main__":
    main()
