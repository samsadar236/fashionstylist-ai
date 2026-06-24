"""
Data layer: load products + outfits, resolve image paths, and build the
co-occurrence compatibility graph from the curated outfits. No model needed.
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C


def load_products(data_dir: Path | None = None) -> pd.DataFrame:
    d = data_dir or C.DATA_DIR
    df = pd.read_csv(d / "products.csv")
    df["coarse_type"] = [C.coarse_type(c, w)
                         for c, w in zip(df["category"], df["wear_type"])]
    df["image_path"] = df["image"].map(lambda p: str(d / p))
    return df


def load_outfits(data_dir: Path | None = None) -> pd.DataFrame:
    d = data_dir or C.DATA_DIR
    return pd.read_csv(d / "outfits.csv")


def product_lookup(products: pd.DataFrame) -> dict[str, dict]:
    return {r["id"]: r for r in products.to_dict("records")}


def outfit_item_ids(row) -> list[str]:
    """All product ids referenced by an outfit row, in role order."""
    return [row[idc] for _, idc in C.ROLE_COLS if pd.notna(row.get(idc))]


class CompatGraph:
    """Co-occurrence graph: which items appear together in curated outfits."""

    def __init__(self, outfits: pd.DataFrame):
        self.partners: dict[str, Counter] = defaultdict(Counter)
        self.pair_count: Counter = Counter()
        self.item_outfits: dict[str, list[str]] = defaultdict(list)
        for _, row in outfits.iterrows():
            ids = outfit_item_ids(row)
            for i in ids:
                self.item_outfits[i].append(row["outfit_id"])
            for a, b in combinations(sorted(set(ids)), 2):
                self.pair_count[frozenset((a, b))] += 1
                self.partners[a][b] += 1
                self.partners[b][a] += 1

    def direct_partners(self, item_id: str) -> Counter:
        """Items that co-occur with item_id in the curated outfits."""
        return self.partners.get(item_id, Counter())

    def known(self, item_id: str) -> bool:
        return item_id in self.partners


if __name__ == "__main__":
    # quick self-test against whatever DATA_DIR points to
    prods = load_products()
    outs = load_outfits()
    g = CompatGraph(outs)
    print(f"products={len(prods)} outfits={len(outs)} "
          f"pairs={len(g.pair_count)} graphed_items={len(g.partners)}")
    print("coarse_type counts:\n", prods["coarse_type"].value_counts().to_string())
