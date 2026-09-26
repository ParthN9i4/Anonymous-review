#!/usr/bin/env bash
# infrastructure/push_results.sh -- ship result artifacts off the H200 as they land.
#
# WHY
#   Results are produced on the cluster; analysis happens elsewhere. Without
#   this, every completed seed has to be relayed by hand, and a session that
#   cannot reach the cluster cannot aggregate anything.
#
# WHY IT IS SAFE TO RUN WHILE AN ARRAY IS STILL RUNNING
#   It never pulls, never merges, and never writes to a tracked source file.
#   A queued array task re-reads core/substitution_ablation.py from disk at
#   task-start, so a pull mid-array would silently split one experiment across
#   two code versions (see infrastructure/RERUN_PLAN.md). A push has no such effect: nothing
#   in the working tree of THIS checkout changes.
#
# HOW IT AVOIDS TOUCHING YOUR CHECKED-OUT BRANCH (fixed 2026-09-11)
#   An earlier version committed directly onto whatever branch was checked
#   out here, then pushed that commit under a different name
#   (`HEAD:refs/heads/results-h200`). The commit's SHA reached results-h200
#   fine, but the LOCAL branch ref gained a commit origin's copy of that same
#   branch never got -- while origin kept advancing from other sessions. The
#   next `git pull` on the training branch failed with "divergent branches",
#   because it had, genuinely, diverged. Reproduced and confirmed 2026-09-11.
#
#   Now the commit is made in a throwaway `git worktree` checked out to
#   results-h200 (or an orphan branch if it does not exist yet), so the
#   branch you have checked out here is never touched -- not staged into,
#   not committed onto, not moved. The worktree is a plain temp directory,
#   deleted on exit whether the script succeeds or fails.
#
# USAGE
#   bash infrastructure/push_results.sh                 # push whatever is on disk now
#   watch -n 900 bash infrastructure/push_results.sh    # or re-run it after each batch lands
#
# Exits 0 with "nothing new" when there is nothing to ship, so it is safe in a
# loop or a cron entry.

set -euo pipefail

BRANCH="${RESULTS_BRANCH:-results-h200}"
REPO_DIR="${REPO_DIR:-$HOME/Private-ViT-FHE}"
cd "$REPO_DIR"

# The training code commit is simply HEAD now: this script no longer commits
# onto the checked-out branch, so HEAD can never be one of ITS commits.
SRC_COMMIT="$(git rev-parse --short HEAD)"
HOST="$(hostname)"

git worktree prune -q 2>/dev/null || true
WORKTREE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/push_results.XXXXXX")"
cleanup() { git worktree remove --force "$WORKTREE_DIR" >/dev/null 2>&1 || rm -rf "$WORKTREE_DIR"; }
trap cleanup EXIT

git fetch -q origin "$BRANCH" 2>/dev/null || true
if git show-ref --verify --quiet "refs/remotes/origin/$BRANCH"; then
  git worktree add -q -B "$BRANCH" "$WORKTREE_DIR" "origin/$BRANCH"
else
  # First push ever: no remote branch to base on. An orphan keeps the results
  # history separate from the training branch's history from the very start.
  git worktree add -q --detach "$WORKTREE_DIR" HEAD
  (cd "$WORKTREE_DIR" && git checkout -q --orphan "$BRANCH" && git reset -q --hard \
     && git commit -q --allow-empty -m "results-h200: root")
fi

# Mirror ONLY the whitelisted paths into the worktree. Never a source file --
# this script must not be able to ship a half-finished code change off the
# cluster by accident, and copying just these two directories makes that
# impossible regardless of what else is dirty in the main checkout.
for pth in results/ablations slurm_logs; do
  mkdir -p "$WORKTREE_DIR/$pth"
  if [ -d "$pth" ]; then
    find "$pth" -maxdepth 1 -type f -exec cp -p -- {} "$WORKTREE_DIR/$pth"/ \;
  fi
done

cd "$WORKTREE_DIR"
git add -A -- results/ablations slurm_logs

if git diff --cached --quiet; then
  echo "nothing new to push (results branch '$BRANCH' is already current)"
  exit 0
fi

echo "staging:"
git diff --cached --name-only | sed 's/^/  /'

git -c user.name="h200-results" -c user.email="noreply@localhost" \
    commit -q -m "results from ${HOST}: artifacts as of $(date -u +%Y-%m-%dT%H:%M:%SZ)

Training code commit (main checkout HEAD at push time): ${SRC_COMMIT}
Pass this to tools/aggregate_seeds.py --source-commit when merging.

Pushed by infrastructure/push_results.sh from a detached worktree; the branch checked out
in the main working directory is never touched."

for attempt in 1 2 3 4; do
  if git push origin "HEAD:refs/heads/${BRANCH}"; then
    echo
    echo "pushed to '${BRANCH}'. Training commit: ${SRC_COMMIT}"
    echo "Merge with:  python tools/aggregate_seeds.py --pattern '<prefix>_seeds*' \\"
    echo "                 --require-seeds --source-commit ${SRC_COMMIT}"
    exit 0
  fi
  delay=$((2 ** attempt))
  echo "push failed (attempt ${attempt}/4); retrying in ${delay}s" >&2
  sleep "${delay}"
done

echo "push failed after 4 attempts. The commit exists in a worktree that is" >&2
echo "about to be removed -- re-run the script; nothing on disk in" >&2
echo "results/ablations/ or slurm_logs/ is touched by this failure." >&2
exit 1
