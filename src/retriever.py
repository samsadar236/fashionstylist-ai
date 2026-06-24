"""
Retrieval (Requirements 3 + 4 backbone).

- OutfitRetriever: NL/occasion query -> nearest curated outfits (gender-filtered),
  returned with their items + the stylist rationale.
- similarity_fn: k nearest catalog items to a given item, used by the
  compatibility engine and the eval harness for similarity-transfer.
- cohesion: mean pairwise embedding similarity of a set of items (a "how well
  do these pieces cohere" score for the UI).

Loads the 68 item vectors into memory at init (trivial) for fast similarity.
"""
from __future__ import annotations

import sys
from pathlib import Path

import chromadb
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C
from embeddings import FashionEncoder

_GENDER = {"male": "men", "men": "men", "man": "men", "boy": "men",
           "female": "women", "women": "women", "woman": "women", "girl": "women"}


class Retriever:
    def __init__(self, encoder: FashionEncoder | None = None):
        self.enc = encoder or FashionEncoder(C.FASHION_CLIP_ID)
        client = chromadb.PersistentClient(path=str(C.CHROMA_DIR))
        self.items = client.get_collection(C.ITEM_COLLECTION)
        self.outfits = client.get_collection(C.OUTFIT_COLLECTION)
        got = self.items.get(include=["embeddings"])
        self.item_ids = got["ids"]
        self.item_mat = np.array(got["embeddings"], dtype=np.float32)
        self._idx = {i: n for n, i in enumerate(self.item_ids)}

    def retrieve_outfits(self, query_text: str, gender: str | None,
                         n: int = C.TOP_OUTFITS) -> list[dict]:
        vec = self.enc.embed_text(query_text).tolist()
        where = None
        g = _GENDER.get((gender or "").lower())
        if g:
            where = {"gender": g}
        res = self.outfits.query(query_embeddings=[vec], n_results=n, where=where)
        out = []
        if res["ids"] and res["ids"][0]:
            for oid, meta, dist in zip(res["ids"][0], res["metadatas"][0],
                                       res["distances"][0]):
                m = dict(meta)
                m["score"] = round(1.0 - float(dist), 4)
                m["item_ids"] = m["item_ids"].split(",") if m.get("item_ids") else []
                out.append(m)
        return out

    def similarity_fn(self, item_id: str, k: int) -> list[tuple[str, float]]:
        if item_id not in self._idx:
            return []
        v = self.item_mat[self._idx[item_id]]
        sims = self.item_mat @ v
        order = np.argsort(-sims)
        out = []
        for j in order:
            cid = self.item_ids[j]
            if cid == item_id:
                continue
            out.append((cid, float(sims[j])))
            if len(out) >= k:
                break
        return out

    def cohesion(self, ids: list[str]) -> float | None:
        """Mean pairwise cosine similarity of the given items' embeddings."""
        vecs = [self.item_mat[self._idx[i]] for i in ids if i in self._idx]
        if len(vecs) < 2:
            return None
        m = np.stack(vecs)
        sims = m @ m.T
        iu = np.triu_indices(len(vecs), k=1)
        return float(sims[iu].mean())
