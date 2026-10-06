"""Render index.html from paper/paper.md: the paper with every figure embedded
as base64, a sticky contents rail generated from the headings, and a banner
linking the PDF, the LaTeX, the code and the companion site.

The <head> carries Highwire Press `citation_*` tags, which is what Google
Scholar reads to index a paper hosted on a personal site. The PDF they point at
must be served from this same site, and the author name must match the PDF's
first page.

    python3 paper/build_paper.py   # first, to produce paper/paper.md
    python3 build_site.py

The page loads no external assets, so opening index.html locally renders it
exactly as GitHub Pages will.
"""
import base64, pathlib, re, subprocess

HERE = pathlib.Path(__file__).resolve().parent

TITLE = "Recursive Axis Conditioning for Diverse Synthetic Data Generation"
AUTHOR = "Halley Young"            # as on the PDF; Scholar matches the two
AUTHOR_CITATION = "Young, Halley"  # surname-first form for citation_author
DATE_SHOWN = "October 2026"
PUB_DATE = "2026/10/06"            # citation_publication_date, YYYY/MM/DD
SITE_URL = "https://thehalleyyoung.github.io/rac/"
PDF_URL = SITE_URL + "paper/paper.pdf"
DOI = "10.5281/zenodo.23197986"                # Zenodo concept DOI: resolves to the latest version
REPO_URL = "https://github.com/thehalleyyoung/rac"
COMPANION_URL = "https://thehalleyyoung.github.io/proxy-embeddings/"
COMPANION_LABEL = "companion paper: Proxy Embeddings"

# One-paragraph summary for search snippets (meta description, og:description).
# Keep it in step with the abstract and with CITATION.cff.
DESCRIPTION = (
    "Recursive Axis Conditioning (RAC) is a loop for generating synthetic "
    "corpora. It asks the generator to name the axes along which its own outputs "
    "vary, ranks those axes, conditions on their most different levels, and "
    "splits an axis into finer ones once it stops producing new items. On "
    "coverage of a held-out human-written reference, RAC places first of twelve "
    "corpora at matched sample size (0.4441 against Alpaca's 0.3722), with a "
    "corpus a twentieth the size of Alpaca's.")

BIBTEX = f"""@misc{{young2026rac,
  title        = {{{{{TITLE}}}}},
  author       = {{{AUTHOR_CITATION}}},
  year         = {{2026}},
  month        = oct,
  publisher    = {{Zenodo}},
  doi          = {{{DOI}}},
  url          = {{https://doi.org/{DOI}}},
  note         = {{Code and data: \\url{{{REPO_URL}}}}}
}}"""


NAV_CSS = """
/* --- section navigation, generated from the paper's own headings --- */
.layout{display:grid;grid-template-columns:15.5rem minmax(0,46rem);gap:2.6rem;
justify-content:center;padding:2.4rem 1.2rem 6rem}
.toc{position:sticky;top:2rem;align-self:start;max-height:calc(100vh - 4rem);
overflow-y:auto;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
font-size:.83rem;line-height:1.45;border-right:1px solid var(--rule);padding-right:1.1rem}
.toc h2{font-size:.72rem;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);
margin:0 0 .7rem;border:0;padding:0}
.toc ol{list-style:none;margin:0;padding:0}
.toc li{margin:.12rem 0}
.toc li.sub{padding-left:.85rem;font-size:.79rem}
.toc a{display:block;padding:.2rem .35rem;border-radius:3px;color:var(--muted);
text-decoration:none;border-left:2px solid transparent}
.toc a:hover{color:var(--fg);background:var(--stripe)}
.toc a.here{color:var(--accent);border-left-color:var(--accent);background:var(--stripe)}
.toc .sec{color:var(--fg)}
main{max-width:none;margin:0;padding:0}
html{scroll-behavior:smooth}
:target{scroll-margin-top:1.5rem}
h2,h3{scroll-margin-top:1.5rem}
@media (max-width:62rem){
  .layout{display:block;max-width:46rem;margin:0 auto}
  .toc{position:static;max-height:none;border-right:0;border-bottom:1px solid var(--rule);
  padding:0 0 1rem;margin-bottom:2rem;columns:2;column-gap:1.6rem}
  .toc li.sub{display:none}
}
@media print{.toc{display:none}.layout{display:block}}
"""


NAV_JS = """
<script>
(function () {
  var links = [].slice.call(document.querySelectorAll('.toc a'));
  var targets = links.map(function (a) {
    return document.getElementById(decodeURIComponent(a.getAttribute('href').slice(1)));
  });
  var current = -1;
  function mark() {
    var best = 0;
    for (var i = 0; i < targets.length; i++) {
      if (targets[i] && targets[i].getBoundingClientRect().top <= 90) best = i;
    }
    if (best === current) return;
    if (current >= 0) links[current].classList.remove('here');
    links[best].classList.add('here');
    current = best;
    var a = links[best];
    var rail = a.closest('.toc');
    if (rail && rail.scrollHeight > rail.clientHeight) {
      var t = a.offsetTop - rail.clientHeight / 2;
      if (Math.abs(rail.scrollTop - t) > 40) rail.scrollTop = t;
    }
  }
  addEventListener('scroll', mark, {passive: true});
  addEventListener('resize', mark, {passive: true});
  addEventListener('load', mark);
  mark();
})();
</script>
"""


def build_toc(body: str) -> str:
    """A contents rail from the rendered headings: sections, and their parts."""
    items = re.findall(r'<h([23]) id="([^"]+)"[^>]*>(.*?)</h[23]>', body, re.S)
    rows = []
    for level, hid, raw in items:
        text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", raw)).strip()
        if text.lower() == "abstract":
            continue
        cls = "sec" if level == "2" else ""
        li = "" if level == "2" else "sub"
        rows.append(f'<li class="{li}"><a class="{cls}" href="#{hid}">{text}</a></li>')
    return ('<nav class="toc" aria-label="Contents"><h2>Contents</h2><ol>'
            + "".join(rows) + "</ol></nav>")


def head_meta() -> str:
    """Scholar's Highwire tags, plus the description and canonical URL."""
    esc = lambda t: t.replace("&", "&amp;").replace('"', "&quot;")
    tags = [
        ("citation_title", TITLE),
        ("citation_author", AUTHOR_CITATION),
        ("citation_publication_date", PUB_DATE),
        ("citation_pdf_url", PDF_URL),
        ("citation_doi", DOI),
        ("citation_abstract_html_url", SITE_URL),
        ("citation_fulltext_html_url", SITE_URL),
        ("citation_language", "en"),
        ("description", DESCRIPTION),
        ("author", AUTHOR),
    ]
    out = [f'<meta name="{k}" content="{esc(v)}">' for k, v in tags]
    out += [f'<meta property="og:title" content="{esc(TITLE)}">',
            f'<meta property="og:description" content="{esc(DESCRIPTION)}">',
            f'<meta property="og:type" content="article">',
            f'<meta property="og:url" content="{SITE_URL}">',
            f'<link rel="canonical" href="{SITE_URL}">']
    return "\n".join(out) + "\n"


def cite_block() -> str:
    """A visible "Cite this paper" section at the foot of the page."""
    esc = lambda t: t.replace("&", "&amp;").replace("<", "&lt;")
    return ('<h2 id="cite">Cite this paper</h2>\n'
            f'<p>Young, H. ({DATE_SHOWN.split()[-1]}). <em>{TITLE}</em>. '
            f'Zenodo. <a href="https://doi.org/{DOI}">https://doi.org/{DOI}</a></p>\n'
            f'<pre><code>{esc(BIBTEX)}</code></pre>\n')


def main():
    css = (HERE / "site.css").read_text()
    tmp = HERE / ".site.md"
    tmp.write_text((HERE / "paper" / "paper.md").read_text())
    body = subprocess.run(
        ["pandoc", "-f", "gfm+tex_math_dollars", "-t", "html5", "--mathml", str(tmp)],
        capture_output=True, text=True, check=True).stdout
    body = body.replace("<table>", '<div class="tw"><table>').replace("</table>", "</table></div>")

    def embed(m):
        src = m.group(1)
        path = HERE / "paper" / src
        if not path.is_file():
            path = HERE / "paper" / "figures" / pathlib.Path(src).name
        if not path.is_file():
            raise SystemExit(f"figure not found for embedding: {src}")
        data = base64.b64encode(path.read_bytes()).decode()
        return m.group(0).replace(src, "data:image/png;base64," + data)

    body = re.sub(r'<img[^>]*src="([^"]+)"', embed, body)
    banner = ('<div class="banner">Paper: <a href="paper/paper.pdf">PDF</a> &middot; '
              '<a href="paper/paper.tex">LaTeX</a> &middot; '
              f'<a href="{REPO_URL}">code &amp; data</a> &middot; '
              f'<a href="{COMPANION_URL}">{COMPANION_LABEL}</a></div>')
    body += cite_block()
    toc = build_toc(body)
    header = (f'<header class="paper-head">\n<h1>{TITLE}</h1>\n'
              f'<p class="byline">{AUTHOR} &middot; {DATE_SHOWN}</p>\n</header>')
    (HERE / "index.html").write_text(
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        f'<title>{TITLE}</title>\n'
        + head_meta() +
        f'<style>{css}{NAV_CSS}</style>\n</head>\n<body>\n'
        f'<div class="layout">\n{toc}\n<main>\n{header}\n{banner}\n{body}\n</main>\n</div>\n'
        f'{NAV_JS}</body>\n</html>\n')
    tmp.unlink()
    n = (HERE / "index.html").read_text().count("data:image/png;base64,")
    print(f"index.html written ({n} figures embedded)")


if __name__ == "__main__":
    main()
