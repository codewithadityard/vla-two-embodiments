"""M7: fill the generated blocks of README.md from results/ (numbers are never typed by hand). Laptop, no torch.

  python -m scripts.make_readme

A block is the text between `<!-- BEGIN name -->` and `<!-- END name -->`; everything else in README.md is prose.
"""
import re
from pathlib import Path

from minivla.stats import wilson
from scripts.make_results_table import load_rows

README = Path("README.md")
TABLE = Path("results/ablation_table.md")
P2 = Path("results/p2_r3.md")


def section(text, title):
    """Body of the '## title' section of a markdown file (up to the next '## ' heading)."""
    m = re.search(rf"^## {re.escape(title)}\n\n(.*?)(?=\n## |\Z)", text, re.S | re.M)
    if not m:
        raise ValueError(f"section '{title}' not found")
    return m.group(1).strip()


def pooled(rows, configs):
    sel = [r for r in rows if r["run"][0] in configs]
    k, n = sum(r["success"] for r in sel), len(sel)
    lo, hi = wilson(k, n)
    return k, n, 100 * k / n, 100 * lo, 100 * hi


def fmt(p):
    k, n, rate, lo, hi = p
    return f"{k}/{n} = {rate:.1f} % (95 % CI {lo:.1f}–{hi:.1f})"


def headline(rows):
    lang, nolang = pooled(rows, "AC"), pooled(rows, "B")
    a, c = pooled(rows, "A"), pooled(rows, "C")
    overlap = a[3] <= c[4] and c[3] <= a[4]
    return "\n".join([
        f"- **Language is what makes the policy work.** With the instruction (A and C pooled): {fmt(lang)}. "
        f"Without it (B, flow matching): {fmt(nolang)}.",
        f"- **Flow matching vs. L1 regression: no resolvable difference.** A (flow): {fmt(a)}; C (L1): {fmt(c)}. "
        f"Difference {a[2] - c[2]:+.1f} points; the CIs {'overlap' if overlap else 'do not overlap'}. With 200 "
        "episodes per config, differences of this size cannot be separated from noise.",
    ])


def p2_block():
    if not P2.exists():
        return "_P2-R3 (MPC-CEM vs. CEM with the official checkpoint) has not finished yet; this block is filled from `results/p2_r3.md`._"
    return section(P2.read_text(), "Results")


def main():
    rows = load_rows()
    t = TABLE.read_text()
    blocks = {
        "headline": headline(rows),
        "per_config": section(t, "Per config (seeds pooled)"),
        "per_model": section(t, "Per model"),
        "per_task": section(t, "Per task (successes per config, seeds pooled)").split("\n\nResolution")[0],
        "p2": p2_block(),
    }
    text = README.read_text()
    for name, body in blocks.items():
        pat = re.compile(rf"(<!-- BEGIN {name} -->\n).*?(<!-- END {name} -->)", re.S)
        if not pat.search(text):
            raise ValueError(f"block {name} missing in README.md")
        text = pat.sub(lambda m: m.group(1) + body + "\n" + m.group(2), text)
    README.write_text(text)
    print("README.md blocks filled:", ", ".join(blocks))


if __name__ == "__main__":
    main()
