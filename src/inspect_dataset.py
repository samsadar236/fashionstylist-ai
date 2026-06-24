"""
Requirement 1: Dataset Analysis. Runs without the model. Writes a markdown
report and prints a summary.

    python src/inspect_dataset.py
"""
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C
from data import CompatGraph, load_outfits, load_products


def main():
    data_dir = None if C.PRODUCTS_CSV.exists() else Path("/mnt/user-data/uploads")
    prods = load_products(data_dir)
    outs = load_outfits(data_dir)
    g = CompatGraph(outs)
    L = ["# Dataset Analysis\n"]

    L += ["## Structure",
          f"- products: **{len(prods)}** | curated outfits: **{len(outs)}**",
          f"- product fields: {list(prods.columns)}",
          f"- outfit fields: {list(outs.columns)}", ""]

    def dist(df, col, title):
        L.append(f"## {title}")
        for v, n in df[col].value_counts(dropna=False).items():
            L.append(f"- {v}: {n}")
        L.append("")

    dist(prods, "gender", "Product gender")
    dist(prods, "wear_type", "Product wear_type")
    dist(prods, "coarse_type", "Derived coarse type")
    dist(prods, "occasion", "Product occasion")
    dist(outs, "occasion", "Outfit occasion")

    L += ["## Palette distribution (outfits)"]
    for v, n in Counter(outs["palette"]).most_common():
        L.append(f"- {v}: {n}")
    L.append("")

    # compatibility graph stats
    appears = Counter()
    for _, r in outs.iterrows():
        for _, idc in C.ROLE_COLS:
            if pd.notna(r.get(idc)):
                appears[r[idc]] += 1
    multi = sum(1 for v in appears.values() if v > 1)
    L += ["## Compatibility graph",
          f"- co-occurrence pairs: {len(g.pair_count)}",
          f"- items used in outfits: {len(appears)} / {len(prods)}",
          f"- items appearing in >1 outfit: {multi} (sparsity → fuse with embeddings)",
          ""]

    # role fill rates
    L += ["## Outfit role fill rate"]
    for name, idc in C.ROLE_COLS:
        L.append(f"- {name}: {int(outs[idc].notna().sum())}/{len(outs)}")
    L.append("")

    # data quality
    L += ["## Data quality notes",
          "- No per-item colour field; colour lives in outfit `palette` and in "
          "product names/descriptions. Colour reasoning is therefore drawn from "
          "the outfit palette / LLM, not a structured field.",
          f"- Missing per column: " + ", ".join(
              f"{c}={int(prods[c].isna().sum())}" for c in prods.columns
              if prods[c].isna().any()) or "- none",
          ""]

    # image presence
    if data_dir is None:
        miss = sum(1 for p in prods["image_path"] if not Path(p).exists())
        L += ["## Images",
              f"- product rows: {len(prods)} | missing image files: {miss}", ""]

    out = Path("docs/dataset_analysis.md")
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print("written to", out)


if __name__ == "__main__":
    main()
