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
