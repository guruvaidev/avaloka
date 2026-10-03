#!/usr/bin/env python3
"""Relative-link checker for the Avaloka docs tree.

Checks every relative markdown link and every relative href/src in HTML:
  - the target file exists
  - if the link carries a #fragment into a markdown file, a heading matches it

Skips: external URLs, mailto:, bare #fragments into the same file (reported
separately), and anything under docs/research/ (owned by another workflow).
"""
import os, re, sys, collections

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
ROOT = os.path.abspath(ROOT)

SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "agents-env", "__pycache__",
             "dist", "build", ".pytest_cache", ".mypy_cache", "site-packages"}
# docs/research is owned by a concurrent workflow -- do not read or report on it
SKIP_PREFIXES = ("docs/research/", "oss/overlay/docs/research/")

MD_LINK   = re.compile(r'(?<!\!)\[([^\]]*)\]\(\s*<?([^)\s>]+)>?(?:\s+"[^"]*")?\s*\)')
MD_IMG    = re.compile(r'\!\[([^\]]*)\]\(\s*<?([^)\s>]+)>?(?:\s+"[^"]*")?\s*\)')
HTML_HREF = re.compile(r'(?:href|src)\s*=\s*["\']([^"\']+)["\']', re.I)
ATX       = re.compile(r'^(#{1,6})\s+(.*?)\s*#*\s*$', re.M)
SETEXT    = re.compile(r'^(?!\s*$)(.+)\n(=+|-+)\s*$', re.M)
HTML_ID   = re.compile(r'\sid\s*=\s*["\']([^"\']+)["\']', re.I)
FENCE     = re.compile(r'^(```|~~~)')

def slug(text):
    # GitHub-flavoured anchor slug
    t = re.sub(r'`([^`]*)`', r'\1', text)                 # strip code ticks
    t = re.sub(r'\*\*([^*]*)\*\*', r'\1', t)              # bold
    t = re.sub(r'\*([^*]*)\*', r'\1', t)                  # italic
    t = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', t)        # inline links
    t = t.strip().lower()
    t = re.sub(r'[^\w\s\-]', '', t, flags=re.UNICODE)
    # GitHub substitutes each space individually, so a heading with an em dash
    # ("A -- B") yields a DOUBLE hyphen. Collapsing runs here reports correct
    # anchors as broken, which is how this checker first lied to us.
    t = t.replace(' ', '-')
    return t

def strip_fences(text):
    """Blank out fenced code blocks so examples don't register as links."""
    out, infence = [], False
    for line in text.split("\n"):
        if FENCE.match(line.strip()):
            infence = not infence
            out.append("")
            continue
        out.append("" if infence else line)
    return "\n".join(out)

def anchors_for(path):
    """Set of valid #fragments in a file."""
    try:
        raw = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return set()
    a = set()
    if path.endswith((".md", ".markdown")):
        body = strip_fences(raw)
        for _, h in ATX.findall(body):
            a.add(slug(h))
        for h, _ in SETEXT.findall(body):
            a.add(slug(h))
    for i in HTML_ID.findall(raw):
        a.add(i)
        a.add(i.lower())
    return a

def rel(p):
    return os.path.relpath(p, ROOT).replace(os.sep, "/")

def collect():
    files = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn.endswith((".md", ".markdown", ".html")):
                p = os.path.join(dirpath, fn)
                if rel(p).startswith(SKIP_PREFIXES):
                    continue
                files.append(p)
    return sorted(files)

anchor_cache = {}
def get_anchors(p):
    if p not in anchor_cache:
        anchor_cache[p] = anchors_for(p)
    return anchor_cache[p]

def main():
    broken = []
    checked = 0
    files = collect()
    for path in files:
        raw = open(path, encoding="utf-8", errors="replace").read()
        body = strip_fences(raw) if path.endswith((".md", ".markdown")) else raw
        targets = []
        if path.endswith((".md", ".markdown")):
            for m in MD_IMG.finditer(body):
                targets.append((m.group(2), body[:m.start()].count("\n") + 1))
            for m in MD_LINK.finditer(body):
                targets.append((m.group(2), body[:m.start()].count("\n") + 1))
        for m in HTML_HREF.finditer(raw):
            targets.append((m.group(1), raw[:m.start()].count("\n") + 1))

        for target, lineno in targets:
            t = target.strip()
            if not t:
                continue
            if "{{" in t or "{%" in t or t.startswith("$"):
                continue          # Go/Jinja template expression, not a link
            low = t.lower()
            if low.startswith(("http://", "https://", "mailto:", "tel:", "data:",
                               "javascript:", "ftp://", "//", "#!")):
                continue
            # bare in-document fragment
            if t.startswith("#"):
                checked += 1
                frag = t[1:]
                if frag and frag.lower() not in {a.lower() for a in get_anchors(path)}:
                    broken.append((rel(path), lineno, t, "no such anchor in this file"))
                continue
            if "#" in t:
                filepart, frag = t.split("#", 1)
            else:
                filepart, frag = t, ""
            filepart = filepart.split("?", 1)[0]
            if not filepart:
                continue
            checked += 1
            # oss/overlay/* is spliced into the generated public tree at its
            # ROOT (oss/manifest.yaml `overlay:` / `splice:`), so its relative
            # links resolve from the repo root, not from oss/overlay/.
            # oss/overlay/* is spliced into the generated public tree at its
            # ROOT (oss/manifest.yaml `overlay:` / `splice:`). In that tree the
            # overlay files sit at root alongside the carried-over ones, so try
            # both: the repo root, then the overlay dir itself.
            if rel(path).startswith("oss/overlay/"):
                cands = [os.path.normpath(os.path.join(ROOT, filepart)),
                         os.path.normpath(os.path.join(ROOT, "oss/overlay", filepart))]
            else:
                cands = [os.path.normpath(os.path.join(os.path.dirname(path), filepart))]
            dest = next((c for c in cands if os.path.exists(c)), cands[0])
            if rel(dest).startswith(SKIP_PREFIXES):
                continue          # concurrent workflow owns these
            if not os.path.exists(dest):
                broken.append((rel(path), lineno, t, "target does not exist"))
                continue
            if frag and dest.endswith((".md", ".markdown", ".html")):
                if frag.lower() not in {a.lower() for a in get_anchors(dest)}:
                    broken.append((rel(path), lineno, t,
                                   f"file exists, no anchor '{frag}'"))

    print(f"files scanned : {len(files)}")
    print(f"links checked : {checked}")
    print(f"broken        : {len(broken)}")
    if broken:
        print()
        bydir = collections.Counter(b[0] for b in broken)
        for f, n in bydir.most_common():
            print(f"  {n:3d}  {f}")
        print()
        for f, ln, t, why in broken:
            print(f"{f}:{ln}: {t}  -- {why}")
    return 1 if broken else 0

sys.exit(main())
