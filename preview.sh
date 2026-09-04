#!/usr/bin/env bash
# Open the site exactly as it will appear on GitHub Pages.
#
# index.html embeds every figure as base64 and pulls in no CDN, so opening the
# file directly renders identically to the published page. The PDF and LaTeX
# links in the banner resolve against the repo, so they work too.
set -euo pipefail
cd "$(dirname "$0")"
[ -f index.html ] || { echo "index.html missing — run: python3 build_site.py"; exit 1; }
case "$(uname -s)" in
  Darwin) open index.html ;;
  Linux)  xdg-open index.html >/dev/null 2>&1 || echo "open: file://$PWD/index.html" ;;
  *)      echo "open: file://$PWD/index.html" ;;
esac
