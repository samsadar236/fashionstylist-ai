"""
Orchestrator: natural-language request -> outfit + reasoning + scores + prices.

Modes (auto-detected): 'occasion' retrieves the nearest curated outfit and
adapts its rationale; 'seed' resolves the user's item to the catalog, runs the
compatibility engine, and assembles a look around it.

A user profile (gender/age/style) can be supplied to fill anything the message
doesn't state.

CLI:
    python src/recommend.py "smart casual for a dinner date, 24M"
    python src/recommend.py "I have a white formal shirt, what goes with it?"
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C
from compatibility import CompatibilityEngine
from data import CompatGraph, load_outfits, load_products
from llm import explain_compat, explain_outfit, parse_intent
from retriever import Retriever


def _price(item: dict):
    v = item.get("price_inr")
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


class Stylist:
    def __init__(self):
        self.prods = load_products()
        self.outs = load_outfits()
        self.retriever = Retriever()
        self.graph = CompatGraph(self.outs)
        self.engine = CompatibilityEngine(
            self.prods, self.graph, sim_fn=self.retriever.similarity_fn)
        self.lookup = {r["id"]: r for r in self.prods.to_dict("records")}

    def _resolve_seed(self, text: str) -> str | None:
        v = self.retriever.enc.embed_text(text)
        sims = self.retriever.item_mat @ v
        return self.retriever.item_ids[int(np.argmax(sims))]

    def recommend(self, msg: str, profile: dict | None = None) -> dict:
        intent = parse_intent(msg)
        if profile:
            for k in ("gender", "style", "age"):
                if not intent.get(k) and profile.get(k):
                    intent[k] = profile[k]

        if intent.get("mode") == "seed" and intent.get("seed_item"):
            return self._seed(msg, intent)
        return self._occasion(msg, intent)

    # --- occasion mode -----------------------------------------------------
    def _occasion(self, msg: str, intent: dict) -> dict:
        query = (f"{intent.get('style','')} {intent.get('occasion','')} outfit "
                 f"for {intent.get('gender') or ''}".strip())
        outfits = self.retriever.retrieve_outfits(query, intent.get("gender"))
        if not outfits:
            return {"mode": "occasion", "intent": intent, "outfit": None,
                    "items": [], "explanation": "No matching outfit found. "
                    "Try a different occasion or clear the gender filter."}
        best = outfits[0]
        recs = [self.lookup[i] for i in best["item_ids"] if i in self.lookup]
        items = [{"id": it["id"], "name": it["name"], "type": it["coarse_type"],
                  "image_path": it["image_path"], "price": _price(it)}
                 for it in recs]
        prices = [i["price"] for i in items if i["price"] is not None]
        return {
            "mode": "occasion", "intent": intent,
            "outfit": {"outfit_id": best["outfit_id"], "theme": best["theme"],
                       "palette": best["palette"], "occasion": best["occasion"],
                       "gender": best["gender"], "match": best["score"]},
            "cohesion": self.retriever.cohesion(best["item_ids"]),
            "items": items,
            "total_price": sum(prices) if prices else None,
            "explanation": explain_outfit(msg, best),
            "alternatives": [o["outfit_id"] for o in outfits[1:]],
        }

    # --- seed mode ---------------------------------------------------------
    def _seed(self, msg: str, intent: dict) -> dict:
        seed_id = self._resolve_seed(intent["seed_item"])
        seed = self.lookup.get(seed_id, {})
        groups_raw = self.engine.recommend(seed_id) if seed_id else {}
        groups = {}
        for t, items in groups_raw.items():
            groups[t] = [{**i, "price": _price(self.lookup.get(i["id"], {}))}
                         for i in items]
        # assembled look = seed + top pick per group
        assembled = [seed_id] + [g[0]["id"] for g in groups.values() if g]
        prices = [_price(self.lookup.get(i, {})) for i in assembled]
        prices = [p for p in prices if p is not None]
        ctx = ", ".join(f"{k}={v}" for k, v in intent.items()
                        if v not in (None, "", False))
        return {
            "mode": "seed", "intent": intent,
            "seed": {"id": seed_id, "name": seed.get("name", ""),
                     "image_path": seed.get("image_path", ""),
                     "type": seed.get("coarse_type", ""), "price": _price(seed)},
            "groups": groups,
            "cohesion": self.retriever.cohesion(assembled),
            "total_price": sum(prices) if prices else None,
            "explanation": explain_compat(seed.get("name", "item"), groups_raw, ctx),
        }


def _print(rec: dict):
    print("\nmode:", rec["mode"], "| intent:", rec["intent"])
    if rec["mode"] == "seed":
        print("seed:", rec["seed"]["name"], f"(₹{rec['seed']['price']})")
        for t, items in rec["groups"].items():
            print(f"  {t}: " + ", ".join(f"{i['name']}" for i in items))
    else:
        o = rec.get("outfit")
        if o:
            print(f"outfit {o['outfit_id']} ({o['theme']}, {o['palette']}) "
                  f"match={o['match']} cohesion={rec.get('cohesion')}")
            for it in rec["items"]:
                print(f"  [{it['type']:9}] {it['name']} (₹{it['price']})")
        print("total ₹", rec.get("total_price"))
    print("why:", rec["explanation"])


if __name__ == "__main__":
    msg = " ".join(sys.argv[1:]) or "casual summer outfit for a 22 year old male"
    _print(Stylist().recommend(msg))
