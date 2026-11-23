#!/usr/bin/env bash
# Generate the public tree from develop-1.6.
#
# The public repository is a product of develop-1.6, not a parallel branch.
# There is no intermediate branch: the generated tree is compared against, and
# published onto, the public repository's own branch.
#
#   scripts/generate-oss.sh             # dry run: generate, check, compare
#   scripts/generate-oss.sh --publish   # build the squash commit; see below
#
# Dry run   generates the tree in a temp dir, runs the invariants and the
#           secret scan, then lists what differs from the published tree
#           ($PUBLISH_REMOTE/$PUBLISH_BASE, default github/main). Without that
#           remote the comparison is skipped, visibly, and the checks still run.
# --publish does the same generation and checks, then builds ONE commit whose
#           only parent is the published tip. It pushes only when
#           PUBLISH_CONFIRM=yes.
#
set -euo pipefail

usage () {
  cat <<'USAGE'
usage: scripts/generate-oss.sh [--publish]

  (no argument)  dry run: generate the public tree, run the invariants and the
                 secret scan, and list what differs from the published tree.
                 Writes nothing and pushes nothing.
  --publish      build one squash commit on top of the published tip and print
                 the push command. Pushes only with PUBLISH_CONFIRM=yes.

environment:
  SRC_BRANCH      branch to generate from          (default develop-1.6)
  REMOTE          remote that holds SRC_BRANCH      (default origin)
  PUBLISH_REMOTE  remote of the public repository   (default github)
  PUBLISH_BRANCH  public branch to publish to       (default main)
  PUBLISH_BASE    published branch to compare with and build on
                                                    (default $PUBLISH_BRANCH)
  OSS_OUT_DIR     also copy the generated tree here once every check passes
                  (must not exist yet)              (default: not kept)
USAGE
}

SRC_BRANCH="${SRC_BRANCH:-develop-1.6}"
REMOTE="${REMOTE:-origin}"
PUBLISH_REMOTE="${PUBLISH_REMOTE:-github}"
PUBLISH_BRANCH="${PUBLISH_BRANCH:-main}"
# Base defaults to the destination, but a new test branch has no tip yet --
# point PUBLISH_BASE at the published branch it should sit on top of.
PUBLISH_BASE="${PUBLISH_BASE:-$PUBLISH_BRANCH}"

case "${1:-}" in
  "" | --publish) ;;
  -h | --help) usage; exit 0 ;;
  --write)
    # Refused before any work is done. --write used to commit the generated
    # tree onto an intermediate branch that sat between the source and the
    # public repository. That branch is retired: nothing read it, it drifted
    # from what was actually published, and it looked like a branch one could
    # target. An old habit should fail here rather than recreate it.
    {
      echo "generate-oss.sh: --write is retired."
      echo "  It committed the generated tree onto an intermediate branch that no longer"
      echo "  exists as a step. There are two modes now:"
      echo "    scripts/generate-oss.sh             dry run, compared against $PUBLISH_REMOTE/$PUBLISH_BASE"
      echo "    scripts/generate-oss.sh --publish   one squash commit on top of $PUBLISH_REMOTE/$PUBLISH_BASE"
    } >&2
    exit 2 ;;
  *)
    echo "generate-oss.sh: unknown argument '$1'" >&2
    usage >&2
    exit 2 ;;
esac

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"
manifest="oss/manifest.yaml"
[ -f "$manifest" ] || { echo "missing $manifest" >&2; exit 1; }

# Read the manifest without a YAML dependency: these are flat lists.
list_of () {  # $1 = top-level key
  awk -v k="$1:" '
    $0 ~ "^"k"$" {inb=1; next}
    inb && /^[a-z_]+:/ {inb=0}
    inb && /^  - path:/ {sub(/^  - path: */,""); gsub(/"/,""); print}
    inb && /^  - [A-Za-z]/ {sub(/^  - */,""); print}
  ' "$manifest"
}

work=$(mktemp -d); trap 'rm -rf "$work"' EXIT
git archive "$REMOTE/$SRC_BRANCH" | tar -x -C "$work"

echo "== excluding =="
python3 - "$work" "$manifest" <<'EXC'
import sys, pathlib, re, shutil, fnmatch
work, mtext = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]).read_text()
blk = re.search(r"^exclude:\n((?:(?![a-z_]+:).*\n)*)", mtext, re.M)
entries, cur = [], None
for line in blk.group(1).split("\n"):
    m = re.match(r"^  - path: *(.*)$", line)
    if m:
        cur = {"path": m.group(1).strip().strip('"'), "keep": []}
        entries.append(cur)
    elif cur is not None:
        k = re.match(r"^    keep_globs: *\[(.*)\]", line)
        if k:
            cur["keep"] = [g.strip().strip('"').strip("'") for g in k.group(1).split(",") if g.strip()]
for e in entries:
    t = work / e["path"]
    if not t.exists():
        print(f'   . {e["path"]} (already absent)'); continue
    if e["keep"]:
        kept = dropped = 0
        for f in sorted(t.rglob("*")):
            if f.is_file():
                if any(fnmatch.fnmatch(f.name, g) for g in e["keep"]):
                    kept += 1
                else:
                    f.unlink(); dropped += 1
        for d in sorted(t.rglob("*"), reverse=True):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
        print(f'   ~ {e["path"]} kept {kept} ({", ".join(e["keep"])}), dropped {dropped}')
    else:
        shutil.rmtree(t) if t.is_dir() else t.unlink()
        print(f'   - {e["path"]}')
EXC

echo "== overlay =="
while read -r f; do
  [ -z "$f" ] && continue
  cp "oss/overlay/$f" "$work/$f"; echo "   + $f"
done < <(list_of overlay)

# Splices: part of a file replaced, the rest left to follow develop-1.6. Each
# entry names a file and exactly one anchor:
#   section: "## X"   the heading line starting with it, up to the next "## "
#   line:    "..."    one line, matched exactly
#   match:   '...'    one line, matched by a regular expression (fullmatch)
#   before:  "..."    insert ahead of one line, matched exactly
# and the replacement as `with:` (one line) or `from:` (a file). An anchor
# that matches no line, or more than one, fails the generation: a silent
# no-op here republishes the internal text.
python3 - "$work" "$manifest" <<'PY'
import sys, pathlib, re
work, mtext = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]).read_text()

def scalar(raw):
    raw = raw.strip()
    if raw[:1] == "'":
        return raw[1:raw.rindex("'")].replace("''", "'")
    if raw[:1] == '"':
        return re.sub(r'\\(.)', r'\1', raw[1:raw.rindex('"')])
    return re.split(r"\s+#", raw, maxsplit=1)[0].strip()

print("== splice ==")
blk = re.search(r"^splice:\n((?:(?![a-z_]+:).*\n)*)", mtext, re.M)
entries, cur = [], None
for line in (blk.group(1).split("\n") if blk else []):
    m = re.match(r"^  - file: *(.*)$", line)
    if m:
        cur = {"file": scalar(m.group(1))}
        entries.append(cur)
        continue
    m = re.match(r"^    ([a-z_]+): *(.*)$", line)
    if m and cur is not None:
        cur[m.group(1)] = scalar(m.group(2))
bad = 0
for e in entries:
    anchors = [k for k in ("section", "line", "match", "before") if k in e]
    label = f'{e["file"]} {anchors[0]}: {e[anchors[0]]}' if anchors else e["file"]
    if len(anchors) != 1 or ("with" in e) == ("from" in e):
        print(f"   FAIL {label} -- needs one anchor and one of with:/from:"); bad += 1; continue
    kind = anchors[0]
    repl = (pathlib.Path(e["from"]).read_text().rstrip("\n").split("\n")
            if "from" in e else [e["with"]])
    p = work / e["file"]
    if not p.is_file():
        print(f"   FAIL {label} -- not in the generated tree"); bad += 1; continue
    lines = p.read_text().split("\n")
    want = e[kind]
    if kind == "section":
        hits = [i for i, l in enumerate(lines) if l.startswith(want)]
    elif kind == "match":
        hits = [i for i, l in enumerate(lines) if re.fullmatch(want, l)]
    else:
        hits = [i for i, l in enumerate(lines) if l == want]
    if len(hits) != 1:
        print(f"   FAIL {label} -- anchor matches {len(hits)} lines, expected 1"); bad += 1; continue
    s = hits[0]
    if kind == "section":
        end = next((i for i, l in enumerate(lines[s+1:], s+1) if l.startswith("## ")), len(lines))
        lines[s:end] = repl + [""]
        note = f"{end-s} lines -> {len(repl)}"
    elif kind == "before":
        lines[s:s] = repl + [""]
        note = f"+{len(repl) + 1} lines"
    else:
        lines[s:s+1] = repl
        note = f"1 line -> {len(repl)}"
    p.write_text("\n".join(lines))
    print(f"   ~ {label} ({note})")
if bad:
    sys.exit(f"   {bad} splice(s) failed -- refusing to write")
PY

# Public text: literal replacements read from a table the public tree does not
# carry (`replace_table:` in the manifest), for text whose old form must not
# be published -- internal change numbers, commit ids -- and so cannot appear
# in this published manifest either. One row per replacement:
#   file<TAB>old<TAB>new[<TAB>count]
# `old` must occur exactly `count` times (default 1), or the run fails. A count
# of `0|N` also allows the text to be absent, for text a pending change on the
# source branch adds or removes; such rows are listed when absent. Files
# are read and written byte-for-byte apart from the replacement, so a CRLF
# file stays CRLF.
python3 - "$work" "$manifest" <<'REP'
import sys, pathlib, re
work, mtext = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]).read_text()
m = re.search(r"^replace_table: *(.+)$", mtext, re.M)
if not m:
    sys.exit(0)
table = pathlib.Path(m.group(1).split(" #")[0].strip().strip('"').strip("'"))
print(f"== public text ({table}) ==")
if not table.is_file():
    sys.exit(f"   FAIL {table} is named by replace_table but is not in this checkout -- refusing to write")
bad, done, absent = 0, {}, []
for n, row in enumerate(table.read_text().split("\n"), 1):
    if not row.strip() or row.startswith("#"):
        continue
    cols = row.split("\t")
    if len(cols) not in (3, 4):
        print(f"   FAIL row {n}: expected file, old, new[, count]"); bad += 1; continue
    fname, old, new = cols[:3]
    # count is N (exactly N), or 0|N (absent, or exactly N) for text that one
    # pending change on the source branch adds or removes. The invariants that
    # follow still check the output, so an absent row cannot let text through.
    spec = cols[3] if len(cols) == 4 else "1"
    optional = spec.startswith("0|")
    want = int(spec[2:] if optional else spec)
    p = work / fname
    if not p.is_file():
        print(f"   FAIL row {n}: {fname} is not in the generated tree"); bad += 1; continue
    with open(p, newline="") as fh:
        text = fh.read()
    got = text.count(old)
    if optional and got == 0:
        absent.append(n); continue
    if got != want:
        print(f"   FAIL row {n}: {fname} -- text occurs {got} times, expected {spec}"); bad += 1; continue
    with open(p, "w", newline="") as fh:
        fh.write(text.replace(old, new))
    done[fname] = done.get(fname, 0) + want
for fname, k in sorted(done.items()):
    print(f"   ~ {fname} ({k} replacement{'s' if k != 1 else ''})")
if absent:
    print(f"   . {len(absent)} optional row(s) not present: {', '.join(map(str, absent))}")
if bad:
    sys.exit(f"   {bad} replacement(s) failed -- refusing to write")
REP

# Invariants. A violation here is the bug this whole mechanism exists to stop.
python3 - "$work" "$manifest" <<'INV'
import sys, pathlib, re
work, manifest = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]).read_text()
print("== invariants ==")

# A needle is a YAML scalar. Stripping the quotes without unescaping left
# `"__version__ = \"0.3.0\""` searching for a backslash that no file carries,
# so that invariant passed whatever avaloka/version.py said.
def scalar(raw):
    raw = raw.strip()
    if raw[:1] == "'":
        return raw[1:raw.rindex("'")].replace("''", "'")
    if raw[:1] == '"':
        return re.sub(r'\\(.)', r'\1', raw[1:raw.rindex('"')])
    return re.split(r"\s+#", raw, maxsplit=1)[0].strip()

bad = 0
for mode in ("present", "absent"):
    blk = re.search(rf"^assert_{mode}:\n((?:(?![a-z_]+:).*\n)*)", manifest, re.M)
    if not blk:
        continue
    fname = None
    for line in blk.group(1).split("\n"):
        if re.match(r"^  \S", line):
            fname = line.strip().rstrip(":")
        elif line.strip().startswith("- ") and fname:
            needle = scalar(line.strip()[2:])
            n = (work / fname).read_text().count(needle)
            ok = (n > 0) if mode == "present" else (n == 0)
            print(f"   {'ok  ' if ok else 'FAIL'} {fname} {mode}: {needle}"
                  + ("" if ok else f"  (found {n})"))
            bad += not ok
if bad:
    sys.exit(f"   {bad} invariant(s) failed -- refusing to write")
INV

# Invariants whose needles must not be published themselves (owner ruling
# 2026-11-02). The manifest names only the published file and a withheld record;
# the needles live in a marked block of that record, read from the same source
# ref as the tree. A missing record or block FAILS generation rather than skipping
# the check, and a failure names the needle by its number, never its text.
python3 - "$work" "$manifest" "$REMOTE/$SRC_BRANCH" <<'WINV'
import subprocess, sys, pathlib, re
work, manifest, src = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]).read_text(), sys.argv[3]
blk = re.search(r"^assert_absent_from_withheld:\n((?:(?![a-z_]+:).*\n)*)", manifest, re.M)
if blk:
    print("== withheld-needle invariants ==")
    entries = re.findall(r"^  - file: *(\S+)\n    needles_from: *(\S+)", blk.group(1), re.M)
    if not entries:
        sys.exit("   FAIL assert_absent_from_withheld lists no entries -- refusing to write")
    bad = 0
    for fname, record in entries:
        shown = subprocess.run(["git", "show", f"{src}:{record}"], capture_output=True, text=True)
        if shown.returncode != 0:
            sys.exit(f"   FAIL {record} is missing from {src}; cannot check {fname} -- refusing to write")
        m = re.search(r"<!-- oss-assert-absent " + re.escape(fname) + r"\n(.*?)\nend-oss-assert-absent -->",
                      shown.stdout, re.S)
        needles = [n.strip() for n in (m.group(1).splitlines() if m else []) if n.strip()]
        if not needles:
            sys.exit(f"   FAIL {record} has no oss-assert-absent block for {fname} -- refusing to write")
        text = (work / fname).read_text()
        for i, needle in enumerate(needles, 1):
            n = text.count(needle)
            print(f"   {'ok  ' if n == 0 else 'FAIL'} {fname} absent: excluded provider #{i} (from {record})"
                  + ("" if n == 0 else f"  (found {n})"))
            bad += n > 0
    if bad:
        sys.exit(f"   {bad} withheld-needle invariant(s) failed -- refusing to write")
WINV

python3 - "$work" "$manifest" <<'PINV'
import sys, pathlib, re
work, mtext = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]).read_text()
blk = re.search(r"^assert_no_paths:\n((?:(?![a-z_]+:).*\n)*)", mtext, re.M)
bad = 0
if blk:
    for line in blk.group(1).split("\n"):
        m = re.match(r"^  - *(\S+)", line)
        if not m:
            continue
        g = m.group(1).strip('"').strip("'")
        # A trailing "**" matches directories only before Python 3.13, so
        # filtering to is_file() left these invariants counting zero and
        # reporting ok on 3.11 and 3.12 -- our whole supported range. They
        # appeared to work only on an interpreter newer than we support.
        # Glob the "/*" form as well so the check is version-independent.
        pats = [g] + ([g.rstrip("/") + "/*"] if g.rstrip("/").endswith("**") else [])
        hits = sorted({str(p.relative_to(work))
                       for pat in pats for p in work.glob(pat) if p.is_file()})
        print(f"   {'FAIL' if hits else 'ok  '} no path matches {g}"
              + (f"  ({len(hits)}: {', '.join(hits[:3])}{'...' if len(hits)>3 else ''})" if hits else ""))
        bad += bool(hits)
if bad:
    sys.exit(f"   {bad} path invariant(s) failed -- refusing to write")
PINV

# Pattern invariants: a regular expression that must match nowhere in the
# files a glob selects. `unless_after` names text that, immediately before a
# match, makes it legitimate (e.g. "GitHub " before a public issue number).
python3 - "$work" "$manifest" <<'PAT'
import sys, pathlib, re
work, mtext = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]).read_text()
blk = re.search(r"^assert_no_match:\n((?:(?![a-z_]+:).*\n)*)", mtext, re.M)
if not blk:
    sys.exit(0)
def scalar(raw):
    raw = raw.strip()
    if raw[:1] == "'":
        return raw[1:raw.rindex("'")].replace("''", "'")
    if raw[:1] == '"':
        return re.sub(r'\\(.)', r'\1', raw[1:raw.rindex('"')])
    return re.split(r"\s+#", raw, maxsplit=1)[0].strip()
entries, cur = [], None
for line in blk.group(1).split("\n"):
    m = re.match(r"^  - glob: *(.*)$", line)
    if m:
        cur = {"glob": scalar(m.group(1))}; entries.append(cur); continue
    m = re.match(r"^    ([a-z_]+): *(.*)$", line)
    if m and cur is not None:
        cur[m.group(1)] = scalar(m.group(2))
bad = 0
for e in entries:
    rx = re.compile(e["pattern"])
    unless = e.get("unless_after", "")
    hits = []
    for p in sorted(work.glob(e["glob"])):
        if not p.is_file():
            continue
        try:
            text = p.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        for n, l in enumerate(text.split("\n"), 1):
            for m in rx.finditer(l):
                if unless and l[:m.start()].endswith(unless):
                    continue
                hits.append(f"{p.relative_to(work)}:{n}")
    print(f"   {'FAIL' if hits else 'ok  '} no match for {e['pattern']} in {e['glob']}"
          + (f"  ({len(hits)}: {', '.join(hits[:5])}{'...' if len(hits) > 5 else ''})" if hits else ""))
    bad += bool(hits)
if bad:
    sys.exit(f"   {bad} pattern invariant(s) failed -- refusing to write")
PAT

python3 - "$work" "$manifest" <<'SEC'
import sys, pathlib, re
work, mtext = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]).read_text()
blk = re.search(r"^assert_no_secrets:\n((?:(?![a-z_]+:).*\n)*)", mtext, re.M)
print("== secret scan ==")
if not blk:
    sys.exit(0)
body = blk.group(1)
mt = re.search(r"min_tail: *(\d+)", body)
min_tail = int(mt.group(1)) if mt else 20
def _pat(raw):
    raw = raw.strip()
    if raw[:1] in "\"'":                      # quoted: take the quoted span, ignore any trailing comment
        q = raw[0]
        return raw[1:raw.index(q, 1)]
    return raw.split("#")[0].strip()          # bare: strip an inline comment
pats = [_pat(m.group(1)) for m in re.finditer(r"^    - *(.+)$", body, re.M)]
pats = [p for p in pats if p]
# Assembled here rather than written into the manifest, so the manifest does not
# itself trip the repo-hygiene scanner that looks for these same headers.
if re.search(r"^  pem_private_keys: *true", body, re.M):
    _b, _pk = "-----BEGIN ", " PRIVATE KEY-----"
    pats += [_b + kind + _pk for kind in ("RSA", "OPENSSH", "EC", "DSA")] + [_b.rstrip() + _pk]
hits, scanned = [], 0
for f in work.rglob("*"):
    if not f.is_file() or ".git" in f.parts:
        continue
    try:
        txt = f.read_text(errors="strict")
    except (UnicodeDecodeError, OSError):
        continue
    scanned += 1
    for pat in pats:
        for m in re.finditer(re.escape(pat) + r"([A-Za-z0-9_\-]*)", txt):
            tail = m.group(1)
            if len(tail) >= min_tail and not re.fullmatch(r"[xX*.]+|(?i:test|fake|dummy|example|your|placeholder|redacted).*", tail):
                hits.append(f"{f.relative_to(work)}: {pat} + {len(tail)} chars")
for pat in pats:
    n = sum(1 for h in hits if pat in h)
    print(f"   {'FAIL' if n else 'ok  '} {pat}" + (f"  ({n})" if n else ""))
print(f"   scanned {scanned} text files, min_tail={min_tail}")
if hits:
    for h in hits[:10]:
        print(f"     {h}")
    sys.exit(f"   {len(hits)} possible secret(s) -- refusing to write")
SEC

# Internal names: words that must not appear in ANY published text file, case
# insensitively -- the internal repository's host and workspace. The manifest
# writes the patterns as regexes with a character class, so the manifest (which
# is itself published) never contains the words it forbids. `allow` phrases are
# exact, case-sensitive and removed before matching, each with its reason.
python3 - "$work" "$manifest" <<'WORDS'
import sys, pathlib, re
work, mtext = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]).read_text()
print("== internal names ==")
blk = re.search(r"^assert_no_words:\n((?:(?![a-z_]+:).*\n)*)", mtext, re.M)
if not blk:
    print("   NOT CHECKED -- this manifest declares no assert_no_words section")
    sys.exit(0)
body, sub, pats, allow = blk.group(1), None, [], []
for line in body.split("\n"):
    s = re.match(r"^  ([a-z_]+):\s*$", line)
    if s:
        sub = s.group(1); continue
    m = re.match(r'^    - *(?:phrase: *)?"([^"]+)"', line)
    if m:
        (pats if sub == "patterns" else allow if sub == "allow" else []).append(m.group(1))
if not pats:
    sys.exit("   FAIL assert_no_words declares no patterns -- the check would pass vacuously")
rx = [re.compile(p, re.I) for p in pats]
hits, scanned = [], 0
for f in sorted(work.rglob("*")):
    if not f.is_file() or ".git" in f.parts:
        continue
    try:
        txt = f.read_text(errors="strict")
    except (UnicodeDecodeError, OSError):
        continue
    scanned += 1
    masked = txt
    for a in allow:
        masked = masked.replace(a, "")
    for r in rx:
        for m in r.finditer(masked):
            line = masked.count("\n", 0, m.start()) + 1
            hits.append((r.pattern, f"{f.relative_to(work)}:{line}"))
for r in rx:
    n = [h for p, h in hits if p == r.pattern]
    print(f"   {'FAIL' if n else 'ok  '} no file matches /{r.pattern}/i" + (f"  ({len(n)})" if n else ""))
print(f"   scanned {scanned} text files, {len(allow)} allowed phrase(s)")
if hits:
    for _, h in hits[:20]:
        print(f"     {h}")
    sys.exit(f"   {len(hits)} internal name(s) in the public tree -- refusing to write")
WORDS
# Keep a copy of the generated tree, to read it or run its own tests against
# exactly what would be published. Only reached once every check has passed.
if [ -n "${OSS_OUT_DIR:-}" ]; then
  [ -e "$OSS_OUT_DIR" ] && { echo "OSS_OUT_DIR $OSS_OUT_DIR already exists; refusing to overwrite" >&2; exit 1; }
  mkdir -p "$(dirname "$OSS_OUT_DIR")"
  cp -a "$work" "$OSS_OUT_DIR"
  echo "== kept generated tree at $OSS_OUT_DIR =="
fi


# No publicly routable IPv4 literal (a cluster dashboard, a load balancer) in
# the generated tree; manifest assert_no_addresses, oss/address_scan.py.
python3 oss/address_scan.py "$work" "$manifest" || { echo "   routable IPv4 address(es) in the public tree -- refusing to write" >&2; exit 1; }


# ---------------------------------------------------------------------------
# Publish: land the generated tree as ONE commit parented on the public branch
# tip. The internal history is never a parent, so nothing that was ever
# force-added into it -- a .env, a key, a dump -- can reach the public repo by
# being reachable from the ref we push. This is the control that a tree-only
# secret scan cannot provide: `git push <internal-branch>` carries every commit
# behind it regardless of how clean the tip tree is.
#
#   PUBLISH_REMOTE=github PUBLISH_BRANCH=main scripts/generate-oss.sh --publish
#
if [ "${1:-}" = "--publish" ]; then
  echo "== publish (squash onto $PUBLISH_REMOTE/$PUBLISH_BASE -> $PUBLISH_BRANCH) =="
  git fetch -q "$PUBLISH_REMOTE" "$PUBLISH_BASE"
  base=$(git rev-parse FETCH_HEAD)
  echo "   base: $PUBLISH_REMOTE/$PUBLISH_BASE at $(git rev-parse --short "$base")"

  idx=$(mktemp -u)
  # -f is required, not optional. .gitignore lists *.csv and app/sample_data/,
  # and those files are tracked on the source branch only because they were
  # force-added. A plain `git add -A` silently omits every such file, which
  # deletes the sample datasets and test fixtures from the public tree.
  GIT_INDEX_FILE="$idx" GIT_DIR="$repo_root/.git" GIT_WORK_TREE="$work" \
    git -C "$work" add -A -f
  tree=$(GIT_INDEX_FILE="$idx" GIT_DIR="$repo_root/.git" git write-tree)
  rm -f "$idx"
  echo "   tree: $(printf %.12s "$tree")"

  if [ "$tree" = "$(git rev-parse "$base^{tree}")" ]; then
    echo "   no change -- $PUBLISH_BRANCH already matches the generated tree"
    exit 0
  fi

  newc=$(git commit-tree "$tree" -p "$base" -m "Sync from $SRC_BRANCH@$(git rev-parse --short "$REMOTE/$SRC_BRANCH")

Generated by scripts/generate-oss.sh from oss/manifest.yaml and squashed onto
the public history. Internal commits are deliberately not parents of this one.")
  echo "   commit: $(git rev-parse --short "$newc") (1 parent: the public tip)"

  # The whole point of squashing. Verify it rather than assume it.
  probes=$(awk '/^assert_unreachable:/{f=1;next} f&&/^[a-z_]+:/{f=0} f&&/^  - /{gsub(/^  - |"/,"");print}' "$manifest")
  [ -z "$probes" ] && { echo "   FAIL manifest declares no assert_unreachable probes"; exit 1; }
  for probe in $probes; do
    n=$(git log "$newc" --oneline -- "$probe" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$n" != "0" ]; then
      echo "   FAIL $probe is reachable from the new commit ($n) -- refusing to push"
      exit 1
    fi
    echo "   ok   $probe unreachable from the published ref"
  done

  # A push whose source ref is empty DELETES the destination branch. That is
  # what `git push <remote> :refs/heads/<branch>` means, and it closed an open
  # pull request once when a caller captured this commit id from the output and
  # the capture came back empty. Refuse rather than emit something unusable.
  case "$newc" in
    *[!0-9a-f]* | "" ) echo "   FAIL refusing to publish: commit id is not a sha ('$newc')"; exit 1 ;;
  esac
  [ "${#newc}" -eq 40 ] || { echo "   FAIL commit id is ${#newc} chars, expected 40"; exit 1; }

  if [ "${PUBLISH_CONFIRM:-}" != "yes" ]; then
    echo "   dry run. Set PUBLISH_CONFIRM=yes to push:"
    echo "     git push $PUBLISH_REMOTE $newc:refs/heads/$PUBLISH_BRANCH"
    exit 0
  fi
  [ -n "$newc" ] || { echo "   FAIL empty commit id at push time"; exit 1; }
  git push "$PUBLISH_REMOTE" "+$newc:refs/heads/$PUBLISH_BRANCH"
  echo "   pushed to $PUBLISH_REMOTE/$PUBLISH_BRANCH"
  exit 0
fi

# ---------------------------------------------------------------------------
# Dry run: compare the generated tree with what is published now -- the same
# base --publish would build on. Nothing is committed and no object is written.
#
# A checkout without the public remote cannot make that comparison. Say so and
# stop there: the generation, invariants and secret scan above have already
# run, and those are what a contributor without publish access needs. An empty
# diff printed in place of a comparison that never happened reads as "nothing
# would change", which is the wrong thing to tell anyone.
echo "== compare vs $PUBLISH_REMOTE/$PUBLISH_BASE (dry run; nothing is written or pushed) =="
generated=$(cd "$work" && find . -type f | wc -l | tr -d ' ')
if ! fetch_err=$(git fetch -q "$PUBLISH_REMOTE" "$PUBLISH_BASE" 2>&1); then
  if git remote get-url "$PUBLISH_REMOTE" >/dev/null 2>&1; then
    why="could not fetch '$PUBLISH_BASE' from '$PUBLISH_REMOTE': $(printf '%s\n' "$fetch_err" | grep -v '^$' | tail -1)"
  else
    why="this checkout has no remote named '$PUBLISH_REMOTE'"
  fi
  echo "   SKIPPED -- $why"
  echo "   The tree was generated and the invariants and secret scan above did run;"
  echo "   only the comparison with the published tree was not made."
  echo "   To compare, add the public repository as a remote, or point"
  echo "   PUBLISH_REMOTE / PUBLISH_BASE at the published branch."
  echo "   generated files: $generated"
  exit 0
fi
base=$(git rev-parse FETCH_HEAD)
echo "   base: $PUBLISH_REMOTE/$PUBLISH_BASE at $(git rev-parse --short "$base")"

# Read the published tree into a throwaway index and diff the generated files
# against it. Untracked files are listed without the ignore rules, for the same
# reason --publish adds with -f: force-added files are part of the tree.
idx=$(mktemp -u)
gitdir=$(git rev-parse --absolute-git-dir)
GIT_INDEX_FILE="$idx" git read-tree "$base"
changes=$(
  { GIT_INDEX_FILE="$idx" GIT_DIR="$gitdir" GIT_WORK_TREE="$work" \
      git -C "$work" -c core.quotePath=false diff --no-renames --name-status
    GIT_INDEX_FILE="$idx" GIT_DIR="$gitdir" GIT_WORK_TREE="$work" \
      git -C "$work" -c core.quotePath=false ls-files --others | sed "s/^/A$(printf '\t')/"
  } | sort -k2
)
rm -f "$idx"
count () { printf '%s\n' "$changes" | grep -c "^$1" || true; }
if [ -z "$changes" ]; then
  echo "   no change -- $PUBLISH_REMOTE/$PUBLISH_BASE already matches the generated tree"
else
  printf '%s\n' "$changes" | sed 's/^/   /'
  echo "   $(count A) added, $(count '[MT]') modified, $(count D) removed"
fi
echo "   generated files: $generated"
echo "   public tree now: $(git ls-tree -r --name-only "$base" | wc -l | tr -d ' ')"
