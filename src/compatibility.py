"""
Outfit Compatibility Engine (Requirement 2).

Given a seed item, return compatible items grouped by complementary type.

Fusion of two signals (justified by measured 16% co-occurrence-only recall):
  1. DIRECT co-occurrence  -- items that appeared with the seed in curated outfits.
     Precise, but only 19/68 items repeat across outfits, so it is sparse.
  2. SIMILARITY-TRANSFER   -- find items visually/semantically similar to the seed
     via FashionCLIP, and borrow THEIR outfit partners. This generalises the
     sparse graph through the embedding space (memory-based CF, visual kernel).

score(candidate) = W_COOCCUR * direct_count_norm + W_SIMILAR * transferred_score

The embedding-similarity function is injected so this module stays testable
without the model (pass sim_fn=None to use co-occurrence only).
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Callable

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C
from data import CompatGraph


def _norm(counter: Counter) -> dict[str, float]:
    if not counter:
        return {}
    m = max(counter.values())
    return {k: v / m for k, v in counter.items()}


class CompatibilityEngine:
    def __init__(self, products: pd.DataFrame, graph: CompatGraph,
                 sim_fn: Callable[[str, int], list[tuple[str, float]]] | None = None):
        """
        sim_fn(item_id, k) -> [(neighbour_id, similarity), ...] for the k most
        similar catalog items (excluding item_id). Optional; if None, only the
        direct co-occurrence signal is used.
        """
        self.products = products
        self.lookup = {r["id"]: r for r in products.to_dict("records")}
        self.graph = graph
        self.sim_fn = sim_fn

    def _transferred_partners(self, seed_id: str) -> Counter:
        """Borrow partners from items similar to the seed."""
        out: Counter = Counter()
        if self.sim_fn is None:
            return out
        for nbr_id, sim in self.sim_fn(seed_id, C.COMPAT_K_SIMILAR):
            for partner, cnt in self.graph.direct_partners(nbr_id).items():
                out[partner] += sim * cnt
        return out

    def recommend(self, seed_id: str) -> dict[str, list[dict]]:
        if seed_id not in self.lookup:
            raise KeyError(f"{seed_id} not in catalog")
        seed = self.lookup[seed_id]
        seed_type = seed["coarse_type"]
        seed_gender = seed.get("gender")
        wanted = set(C.COMPLEMENTS.get(seed_type, []))

        direct = _norm(self.graph.direct_partners(seed_id))
        transfer = _norm(self._transferred_partners(seed_id))

        scored: Counter = Counter()
        for cid in set(direct) | set(transfer):
            if cid == seed_id:
                continue
            scored[cid] = (C.W_COOCCUR * direct.get(cid, 0.0)
                           + C.W_SIMILAR * transfer.get(cid, 0.0))

        # group by complementary type, keep top-N each, gender-consistent
        by_type: dict[str, list[dict]] = {t: [] for t in wanted}
        for cid, s in scored.most_common():
            item = self.lookup.get(cid)
            if not item:
                continue
            if seed_gender and item.get("gender") != seed_gender:
                continue  # don't mix men's and women's pieces in one look
            t = item["coarse_type"]
            if t in wanted and len(by_type[t]) < C.COMPAT_TOPN:
                by_type[t].append({
                    "id": cid, "name": item["name"], "type": t,
                    "category": item["category"], "score": round(float(s), 4),
                    "image_path": item["image_path"],
                })
        return {t: v for t, v in by_type.items() if v}


if __name__ == "__main__":
    from data import load_outfits, load_products
    up = Path("/mnt/user-data/uploads")
    D = up if up.exists() else None          # None -> uses config.DATA_DIR
    prods, outs = load_products(D), load_outfits(D)
    eng = CompatibilityEngine(prods, CompatGraph(outs), sim_fn=None)
    seed = prods[prods["coarse_type"] == "top"].iloc[0]["id"]
    print("seed:", seed, "(", eng.lookup[seed]["name"], ")")
    for t, items in eng.recommend(seed).items():
        print(f"  {t}: " + ", ".join(f"{i['name']}({i['score']})" for i in items))