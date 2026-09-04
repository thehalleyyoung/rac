"""Render index.html from paper/paper.md: the paper with every figure embedded
as base64, a sticky contents rail generated from the headings, and a banner
linking the PDF, the LaTeX, the code and the companion site.

    python3 paper/build_paper.py   # first, to produce paper/paper.md
    python3 build_site.py

The page loads no external assets, so opening index.html locally renders it
exactly as GitHub Pages will.
"""
import base64, pathlib, re, subprocess

HERE = pathlib.Path(__file__).resolve().parent

TITLE = "Recursive Axis Conditioning for Diverse Synthetic Data Generation"
REPO_URL = "https://github.com/thehalleyyoung/rac"
COMPANION_URL = "https://thehalleyyoung.github.io/proxy-embeddings/"
COMPANION_LABEL = "companion paper: Proxy Embeddings"

LAND = f"""
# {TITLE}

---
"""


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


def main():
    css = (HERE / "site.css").read_text()
    tmp = HERE / ".site.md"
    tmp.write_text(LAND + (HERE / "paper" / "paper.md").read_text())
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
    toc = build_toc(body)
    (HERE / "index.html").write_text(
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        f'<title>{TITLE}</title>\n'
        f'<style>{css}{NAV_CSS}</style>\n</head>\n<body>\n'
        f'<div class="layout">\n{toc}\n<main>\n{banner}\n{body}\n</main>\n</div>\n'
        f'{NAV_JS}</body>\n</html>\n')
    tmp.unlink()
    n = (HERE / "index.html").read_text().count("data:image/png;base64,")
    print(f"index.html written ({n} figures embedded)")


if __name__ == "__main__":
    main()
