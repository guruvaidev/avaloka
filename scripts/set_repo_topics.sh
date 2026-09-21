#!/usr/bin/env bash
#
# Set the GitHub topics and description for the repository.
#
# Topics are the main thing driving discovery on GitHub: they power
# github.com/topics/<name> browsing, they are weighted in search, and they are
# how someone looking for "an agentic data-science tool" finds one. They cannot
# live in a file in the repo -- they are repository metadata -- so this script
# is the version-controlled record of what they should be.
#
#   ./scripts/set_repo_topics.sh                    # needs `gh auth login`
#   ./scripts/set_repo_topics.sh --dry-run
#   REPO=owner/name ./scripts/set_repo_topics.sh
#
# GitHub allows at most 20 topics, lowercase, digits and hyphens, <= 50 chars.
# We use all 20: there is no cost to an accurate one, and an unused slot is a
# search someone does not find you in.
set -euo pipefail

REPO="${REPO:-guruvaidev/avaloka}"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# Ordered roughly by how much traffic each one carries. The first five are
# high-volume browse topics; the middle block is what Avaloka actually is; the
# technology tags reach people searching for the stack rather than the problem.
TOPICS=(
  # what it is -- the highest-traffic topics on GitHub
  data-science machine-learning llm ai-agents agentic-ai
  # what it does
  automl mlops data-analysis data-engineering etl data-quality
  # how it is built -- finds people searching for the stack
  langgraph langchain ray python kubernetes fastapi
  # how you use it
  cli multi-agent analytics
)

DESCRIPTION="An agentic team of data scientists that takes any dataset from raw data to model inference — profile, plan, code, validate, train and serve, with every claim checked against the computed evidence."

HOMEPAGE="https://avaloka.ai"

if [[ "${#TOPICS[@]}" -gt 20 ]]; then
  echo "GitHub allows at most 20 topics; this list has ${#TOPICS[@]}." >&2
  exit 1
fi
for t in "${TOPICS[@]}"; do
  [[ "$t" =~ ^[a-z0-9][a-z0-9-]{0,49}$ ]] || { echo "invalid topic: $t" >&2; exit 1; }
done

printf 'repository : %s\n' "$REPO"
printf 'topics (%2d): %s\n' "${#TOPICS[@]}" "${TOPICS[*]}"
printf 'homepage   : %s\n' "$HOMEPAGE"
printf 'description: %s\n' "$DESCRIPTION"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo
  echo "(dry run -- nothing was changed)"
  exit 0
fi

# Two ways in, because not everyone has gh installed. Either needs a token with
# the `repo` scope (`public_repo` is enough for a public repository).
if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  # topics REPLACE the existing set: this file is the source of truth, not an
  # addition to whatever happens to be there.
  gh api -X PUT "repos/$REPO/topics" \
    -H "Accept: application/vnd.github+json" \
    $(printf -- '-f names[]=%s ' "${TOPICS[@]}") >/dev/null
  gh repo edit "$REPO" --description "$DESCRIPTION" --homepage "$HOMEPAGE" >/dev/null
  echo
  echo "Done (via gh). Verify at https://github.com/$REPO"
  exit 0
fi

if [[ -n "${GITHUB_TOKEN:-}" ]]; then
  names=$(printf '"%s",' "${TOPICS[@]}"); names="[${names%,}]"
  curl -sS -X PUT "https://api.github.com/repos/$REPO/topics" \
    -H "Authorization: Bearer $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github+json" \
    -d "{\"names\": $names}" >/dev/null
  curl -sS -X PATCH "https://api.github.com/repos/$REPO" \
    -H "Authorization: Bearer $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github+json" \
    -d "$(printf '{"description": %s, "homepage": %s}' \
           "$(printf '%s' "$DESCRIPTION" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')" \
           "$(printf '%s' "$HOMEPAGE" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')")" >/dev/null
  echo
  echo "Done (via the API). Verify at https://github.com/$REPO"
  exit 0
fi

cat >&2 <<'MSG'

No way to authenticate to GitHub. Pick one:

  1. gh CLI      brew install gh && gh auth login   then re-run this script
  2. A token     export GITHUB_TOKEN=ghp_...        then re-run this script
                 (needs the `public_repo` scope for a public repository)
  3. By hand     open the repository on github.com, click the gear beside
                 "About" on the right, and paste the topics printed above

Topics are repository metadata, not repository content -- nothing committed to
the repo can set them, which is why this script exists rather than a config file.
MSG
exit 1
