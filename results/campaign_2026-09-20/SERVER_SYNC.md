# Review this evidence on the H200 without changing the active checkout

This GitHub branch records the 20 September research evidence. It does not contain checkpoint bytes, CIFAR data, full prediction arrays or the large campaign archives. It does not replace an active Slurm experiment. Review the branch in a separate worktree:

```bash
cd "$HOME/Private-ViT-FHE"
git status --short
git rev-parse HEAD
mkdir -p "$HOME/pvit_results/repo_provenance"
git diff --binary > "$HOME/pvit_results/repo_provenance/worktree_before_review.patch"
git ls-files --others --exclude-standard > "$HOME/pvit_results/repo_provenance/untracked_before_review.txt"
git fetch origin research/private-vit-evidence-20260920
git worktree add --detach "$HOME/Private-ViT-FHE-evidence-review" FETCH_HEAD
```

Inspect `research/2026-09-20-private-vit/README.md`, `CLAIMS.md`, and `source_final/README.md` inside the review worktree. The active `~/Private-ViT-FHE` checkout and its ongoing Slurm runs remain untouched. Before applying code changes to it, compare its tracked changes and untracked files against the captured provenance and independently verify input/checkpoint paths.

The separately developed `paper/` figures and manuscript are on a different research branch as of this snapshot. Their figure scripts and split definitions require review before the draft uses their graphics or numbers. Neither this branch nor the named GitHub account is suitable for an anonymous submission link.
