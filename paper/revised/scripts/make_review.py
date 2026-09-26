"""Build the paragraph-by-paragraph review from paper.tex markers and the notes file.

    python paper/revised/scripts/make_review.py

Every '% [Px.y]' marker in paper.tex must have a '## Px.y' section in notes.md, and vice versa.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "paper" / "revised" / "paper.tex"
DIR = ROOT / "docs" / "submission" / "revision_2026-09-25"
STOP = re.compile(r"^\\(section|subsection|begin\{figure|begin\{table|input|bibliography)")


def paragraphs():
    out, key, buf = {}, None, []
    for line in PAPER.read_text().splitlines():
        m = re.match(r"^% \[(P[\d.]+)\]$", line)
        if m or (key and STOP.match(line)):
            if key:
                out[key] = "\n".join(buf).strip()
            key, buf = (m.group(1), []) if m else (None, [])
            continue
        if key:
            buf.append(line)
    if key:
        out[key] = "\n".join(buf).strip()
    return out


def notes():
    text = (DIR / "notes.md").read_text()
    parts = re.split(r"^## (.+)$", text, flags=re.M)
    return {parts[i].strip(): parts[i + 1].strip() for i in range(1, len(parts), 2)}


def main():
    paras, note = paragraphs(), notes()
    missing = sorted(set(paras) - set(note))
    orphan = sorted(k for k in note if k.startswith("P") and k not in paras)
    if missing or orphan:
        raise SystemExit(f"notes missing for {missing}; notes without paragraph {orphan}")
    body = [(DIR / "header.md").read_text().rstrip(), ""]
    for key, text in paras.items():
        body += [f"### {key}", "", "**Revised paragraph**", "", "```latex", text, "```", "",
                 "**Notes**", "", note[key], ""]
    body += ["## 6. Figure and table notes", ""]
    for key, text in note.items():
        if not key.startswith("P"):
            body += [f"### {key}", "", text, ""]
    out = DIR / "PAPER_REVISION_REVIEW.md"
    out.write_text("\n".join(body))
    print(out, len(paras), "paragraphs")


if __name__ == "__main__":
    main()
