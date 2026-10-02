#!/usr/bin/env bash
# Generate the public oss/1.6 tree from develop-1.6.
#
# oss/1.6 is a product of develop-1.6, not a parallel branch. Run this after
# merging to develop-1.6; review the diff; push.
#
#   scripts/generate-oss.sh                 # dry run, prints the diff
#   scripts/generate-oss.sh --write         # commit onto the oss branch
#
set -euo pipefail

SRC_BRANCH="${SRC_BRANCH:-develop-1.6}"
DST_BRANCH="${DST_BRANCH:-oss/1.6}"
REMOTE="${REMOTE:-origin}"
WRITE=0
case "${1:-}" in --write) WRITE=1 ;; esac

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

# Splice the public quickstart in place of develop's.
python3 - "$work" <<'PY'
import sys, pathlib, re
work = pathlib.Path(sys.argv[1])
repl = pathlib.Path("oss/overlay/README.quickstart.md").read_text().rstrip("\n").split("\n")
p = work / "README.md"; lines = p.read_text().split("\n")
s = next(i for i, l in enumerate(lines) if l.startswith("## Quickstart"))
e = next(i for i, l in enumerate(lines[s+1:], s+1) if l.startswith("## "))
p.write_text("\n".join(lines[:s] + repl + [""] + lines[e:]))
print(f"== splice ==\n   ~ README.md ## Quickstart ({e-s} lines -> {len(repl)})")
PY

# Invariants. A violation here is the bug this whole mechanism exists to stop.
python3 - "$work" "$manifest" <<'INV'
import sys, pathlib, re
work, manifest = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]).read_text()
print("== invariants ==")
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
            needle = line.strip()[2:].strip().strip('"')
            n = (work / fname).read_text().count(needle)
            ok = (n > 0) if mode == "present" else (n == 0)
            print(f"   {'ok  ' if ok else 'FAIL'} {fname} {mode}: {needle}"
                  + ("" if ok else f"  (found {n})"))
            bad += not ok
if bad:
    sys.exit(f"   {bad} invariant(s) failed -- refusing to write")
INV

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
        hits = [str(p.relative_to(work)) for p in work.glob(g) if p.is_file()]
        print(f"   {'FAIL' if hits else 'ok  '} no path matches {g}"
              + (f"  ({len(hits)}: {', '.join(hits[:3])}{'...' if len(hits)>3 else ''})" if hits else ""))
        bad += bool(hits)
if bad:
    sys.exit(f"   {bad} path invariant(s) failed -- refusing to write")
PINV

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
  PUBLISH_REMOTE="${PUBLISH_REMOTE:-github}"
  PUBLISH_BRANCH="${PUBLISH_BRANCH:-main}"
  # Base defaults to the destination, but a new test branch has no tip yet --
  # point PUBLISH_BASE at the published branch it should sit on top of.
  PUBLISH_BASE="${PUBLISH_BASE:-$PUBLISH_BRANCH}"
  echo "== publish (squash onto $PUBLISH_REMOTE/$PUBLISH_BASE -> $PUBLISH_BRANCH) =="
  git fetch -q "$PUBLISH_REMOTE" "$PUBLISH_BASE"
  base=$(git rev-parse FETCH_HEAD)
  echo "   base: $PUBLISH_REMOTE/$PUBLISH_BASE at $(git rev-parse --short "$base")"

  idx=$(mktemp -u)
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

  if [ "${PUBLISH_CONFIRM:-}" != "yes" ]; then
    echo "   dry run. Set PUBLISH_CONFIRM=yes to push:"
    echo "     git push $PUBLISH_REMOTE $newc:refs/heads/$PUBLISH_BRANCH"
    exit 0
  fi
  git push "$PUBLISH_REMOTE" "$newc:refs/heads/$PUBLISH_BRANCH"
  echo "   pushed to $PUBLISH_REMOTE/$PUBLISH_BRANCH"
  exit 0
fi

if [ "$WRITE" -eq 0 ]; then
  echo "== diff vs $REMOTE/$DST_BRANCH (dry run; pass --write to commit) =="
  git --no-index --no-pager diff --stat \
    <(git ls-tree -r --name-only "$REMOTE/$DST_BRANCH") \
    <(cd "$work" && find . -type f | sed 's|^\./||' | sort) 2>/dev/null || true
  ( cd "$work" && find . -type f | sed 's|^\./||' | sort ) > "$work/../gen.txt"
  echo "   generated files: $(wc -l < "$work/../gen.txt" | tr -d ' ')"
  echo "   public tree now: $(git ls-tree -r --name-only "$REMOTE/$DST_BRANCH" | wc -l | tr -d ' ')"
  exit 0
fi


# Land the generated tree on a branch so it can be reviewed as a diff before it
# reaches the public one. GEN_BRANCH overrides the name.
GEN_BRANCH="${GEN_BRANCH:-oss-sync/generated}"
tmpwt=$(mktemp -d)
git worktree add -q --detach "$tmpwt" "$REMOTE/$DST_BRANCH"
( cd "$tmpwt" && git rm -rq --cached . >/dev/null && find . -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} + )
( cd "$work" && tar -c . ) | tar -x -C "$tmpwt"
src_sha=$(git rev-parse --short "$REMOTE/$SRC_BRANCH")
( cd "$tmpwt"
  # -f is required, not optional. .gitignore lists *.csv and app/sample_data/,
  # and those files are tracked on the source branch only because they were
  # force-added. A plain `git add -A` silently omits every such file, which
  # deletes the sample datasets and test fixtures from the public tree.
  git add -A -f
  if git diff --cached --quiet; then
    echo "no change -- $DST_BRANCH already matches $SRC_BRANCH"
  else
    git commit -q -m "chore(oss): regenerate from $SRC_BRANCH@$src_sha

Generated by scripts/generate-oss.sh from oss/manifest.yaml.
Do not edit $DST_BRANCH directly -- the next run overwrites it."
    git branch -f "$GEN_BRANCH" HEAD
    echo "== wrote =="
    echo "   branch $GEN_BRANCH at $(git rev-parse --short HEAD)"
    echo "   files changed vs $DST_BRANCH: $(git diff --name-only "$REMOTE/$DST_BRANCH" HEAD | wc -l | tr -d ' ')"
    echo "   review:  git diff $REMOTE/$DST_BRANCH $GEN_BRANCH"
    echo "   publish: git push $REMOTE $GEN_BRANCH:$DST_BRANCH"
  fi )
git worktree remove --force "$tmpwt"
