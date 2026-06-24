"""
Evaluation harness: leave-one-out recall against the 25 ground-truth outfits.

For each outfit, hold out one item, predict partners from the remaining items
using ONLY the other 24 outfits (no leakage), and check whether the held-out
item is recovered in the top-k. Pass sim_fn to include FashionCLIP
similarity-transfer; without it you get the co-occurrence-only baseline.

    python src/eval.py            # co-occurrence baseline (no model)
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Callable

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C
from data import CompatGraph, load_outfits, load_products, outfit_item_ids


def _predict(rest_ids: list[str], graph: CompatGraph,
             sim_fn: Callable | None) -> list[str]:
    scores: Counter = Counter()
    for rid in rest_ids:
        for p, c in graph.direct_partners(rid).items():
            scores[p] += C.W_COOCCUR * c
        if sim_fn:
            for nbr, sim in sim_fn(rid, C.COMPAT_K_SIMILAR):
                for p, c in graph.direct_partners(nbr).items():
                    scores[p] += C.W_SIMILAR * sim * c
    return [i for i, _ in scores.most_common() if i not in rest_ids]


def leave_one_out(outfits: pd.DataFrame, ks=(3, 5),
                  sim_fn: Callable | None = None) -> dict[int, float]:
    rows = outfits.to_dict("records")
    hits = {k: 0 for k in ks}
    total = 0
    for held_row in rows:
        ids = outfit_item_ids(held_row)
        if len(ids) < 2:
            continue
        others = pd.DataFrame([r for r in rows
                               if r["outfit_id"] != held_row["outfit_id"]])
        graph = CompatGraph(others)
        for held in ids:
            rest = [x for x in ids if x != held]
            ranked = _predict(rest, graph, sim_fn)
            total += 1
            for k in ks:
                hits[k] += held in ranked[:k]
    return {k: hits[k] / total for k in ks}


if __name__ == "__main__":
    D = Path("/mnt/user-data/uploads") if not C.OUTFITS_CSV.exists() else None
    outs = load_outfits(D)
    res = leave_one_out(outs, sim_fn=None)
    print("Co-occurrence-only baseline:")
    for k, v in res.items():
        print(f"  recall@{k}: {v:.0%}")
    print("\nPass a FashionCLIP sim_fn (see retriever.similarity_fn) to get the "
          "fused number -- expected to beat this baseline.")
