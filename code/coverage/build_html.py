"""Generate paper.html from PAPER.md + figures/*.png.

Single source of truth: the HTML is rendered from PAPER.md, so a change to
the paper or a regenerated figure only requires rerunning this script.
Figures are embedded as base64 data URIs (the page is fully standalone) at
the points marked in PAPER.md by comment lines of the form

    <!--FIG:fig_name.png|caption text-->

The converter supports the markdown subset PAPER.md actually uses: ATX
headers, paragraphs, **bold** / *italic* / `code`, pipe tables, unordered
and ordered lists, blockquotes, and horizontal rules.
"""
from __future__ import annotations

import base64
import html
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"

CSS = """
:root {
  --bg: #ffffff; --fg: #1a1a1a; --muted: #555; --rule: #d8d8d8;
  --accent: #14507a; --code-bg: #f2f2f2; --th-bg: #f5f5f5;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14161a; --fg: #e4e4e4; --muted: #a5a5a5; --rule: #3a3d42;
    --accent: #7db4dc; --code-bg: #22252a; --th-bg: #1d2025;
  }
  img.figure { background: #ffffff; border-radius: 4px; }
}
html { background: var(--bg); }
body {
  background: var(--bg); color: var(--fg);
  font: 17px/1.65 Georgia, 'Times New Roman', serif;
  max-width: 46rem; margin: 0 auto; padding: 2.5rem 1.2rem 5rem;
}
h1 { font-size: 1.75rem; line-height: 1.25; }
h2 { font-size: 1.32rem; margin-top: 2.4rem; border-bottom: 1px solid var(--rule);
     padding-bottom: 0.25rem; }
h3 { font-size: 1.08rem; margin-top: 1.8rem; }
a { color: var(--accent); }
code { font: 0.86em/1.4 ui-monospace, 'SF Mono', Menlo, monospace;
       background: var(--code-bg); padding: 0.1em 0.3em; border-radius: 3px; }
blockquote { border-left: 3px solid var(--accent); margin: 1rem 0;
             padding: 0.05rem 1rem; color: var(--muted); }
blockquote strong { color: var(--fg); }
table { border-collapse: collapse; margin: 1.2rem 0; font-size: 0.86em;
        display: block; overflow-x: auto; }
th, td { border: 1px solid var(--rule); padding: 0.35rem 0.6rem; text-align: left; }
th { background: var(--th-bg); }
hr { border: none; border-top: 1px solid var(--rule); margin: 2.5rem 0; }
img.figure { max-width: 100%; height: auto; margin: 1.4rem 0 0.3rem; }
figure { margin: 1.6rem 0; }
figcaption { font-size: 0.85em; color: var(--muted); font-style: italic; }
"""


def inline_md(s: str) -> str:
    s = html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<em>\1</em>", s)
    return s


def figure_html(name: str, caption: str) -> str:
    path = FIG / name
    if not path.exists():
        return f"<p><em>[figure {html.escape(name)} not found]</em></p>"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return (f'<figure><img class="figure" alt="{html.escape(caption)}" '
            f'src="data:image/png;base64,{b64}">'
            f"<figcaption>{inline_md(caption)}</figcaption></figure>")


def table_html(rows: list[str]) -> str:
    def cells(row):
        return [c.strip() for c in row.strip().strip("|").split("|")]
    out = ["<table>"]
    header = cells(rows[0])
    out.append("<thead><tr>" + "".join(f"<th>{inline_md(c)}</th>" for c in header)
               + "</tr></thead><tbody>")
    for row in rows[2:]:
        out.append("<tr>" + "".join(f"<td>{inline_md(c)}</td>" for c in cells(row))
                   + "</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def md_to_html(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    para: list[str] = []
    listbuf: list[str] = []
    listtag = ""
    quote: list[str] = []

    def flush_para():
        if para:
            out.append(f"<p>{inline_md(' '.join(para))}</p>")
            para.clear()

    def flush_list():
        nonlocal listtag
        if listbuf:
            out.append(f"<{listtag}>" + "".join(f"<li>{inline_md(x)}</li>" for x in listbuf)
                       + f"</{listtag}>")
            listbuf.clear()
            listtag = ""

    def flush_quote():
        if quote:
            out.append(f"<blockquote><p>{inline_md(' '.join(quote))}</p></blockquote>")
            quote.clear()

    while i < len(lines):
        ln = lines[i]
        fig = re.match(r"\s*<!--FIG:([^|]+)\|(.*?)-->\s*$", ln)
        if fig:
            flush_para(); flush_list(); flush_quote()
            out.append(figure_html(fig.group(1).strip(), fig.group(2).strip()))
            i += 1
            continue
        if ln.startswith("|") and i + 1 < len(lines) and \
                re.match(r"^\|[\s\-|:]+\|?\s*$", lines[i + 1]):
            flush_para(); flush_list(); flush_quote()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i]); i += 1
            out.append(table_html(rows))
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", ln)
        if m:
            flush_para(); flush_list(); flush_quote()
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{inline_md(m.group(2))}</h{lvl}>")
            i += 1
            continue
        if re.match(r"^\s*---+\s*$", ln):
            flush_para(); flush_list(); flush_quote()
            out.append("<hr>")
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
            tag = "ul" if mli else "ol"
            if listtag and listtag != tag:
                flush_list()
            listtag = tag
            listbuf.append((mli or moi).group(1))
            # absorb hanging continuation lines
            j = i + 1
            while j < len(lines) and lines[j].startswith("  ") and lines[j].strip() \
                    and not re.match(r"^\s*[-*]\s+|^\s*\d+\.\s+", lines[j]):
                listbuf[-1] += " " + lines[j].strip()
                j += 1
            i = j
            continue
        if not ln.strip():
            flush_para(); flush_list(); flush_quote()
            i += 1
            continue
        flush_list(); flush_quote()
        para.append(ln.strip())
        i += 1
    flush_para(); flush_list(); flush_quote()
    return "\n".join(out)


def main():
    md = (HERE / "PAPER.md").read_text()
    title_m = re.match(r"^#\s+(.+)$", md.splitlines()[0])
    title = title_m.group(1) if title_m else "Coverage at a Finite Budget"
    body = md_to_html(md)
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{CSS}</style>
</head>
<body>
{body}
</body>
</html>
"""
    (HERE / "paper.html").write_text(page)
    print(f"wrote paper.html ({len(page) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
