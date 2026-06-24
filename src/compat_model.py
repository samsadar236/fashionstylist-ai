"""
OPTIONAL / EXPERIMENTAL (ML-depth bonus): a learned pairwise compatibility
scorer. Positive pairs = items that co-occur in the 25 curated outfits;
negatives = sampled non-co-occurring pairs. Features come from FashionCLIP
embeddings. Trains a small logistic-regression ranker and reports cross-
validated AUC.

Honest caveat: with 124 positive pairs this is tiny and will overfit — it is a
demonstration of the *method* (pairwise ranking on multimodal features), not a
production model. Run AFTER build_index.py:

    python src/compat_model.py
"""
import sys
from itertools import combinations
from pathlib import Path

import chromadb
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C
from data import CompatGraph, load_outfits


def _pair_features(ea: np.ndarray, eb: np.ndarray) -> np.ndarray:
    # symmetric, compact: elementwise product + |difference| summary stats
    prod = ea * eb
    diff = np.abs(ea - eb)
    return np.concatenate([prod, [diff.mean(), diff.max(), float(ea @ eb)]])


def main():
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
    except Exception:
        sys.exit("pip install scikit-learn to run this optional module")

    client = chromadb.PersistentClient(path=str(C.CHROMA_DIR))
    got = client.get_collection(C.ITEM_COLLECTION).get(include=["embeddings"])
    emb = {i: np.asarray(v, dtype=np.float32)
           for i, v in zip(got["ids"], got["embeddings"])}
    ids = list(emb)

    g = CompatGraph(load_outfits())
    pos = [tuple(p) for p in g.pair_count]
    pos_set = set(g.pair_count)

    rng = np.random.default_rng(C.__dict__.get("RANDOM_SEED", 42))
    all_pairs = list(combinations(ids, 2))
    negs = []
    while len(negs) < len(pos):
        a, b = all_pairs[rng.integers(len(all_pairs))]
        if frozenset((a, b)) not in pos_set:
            negs.append((a, b))

    X, y = [], []
    for a, b in pos:
        if a in emb and b in emb:
            X.append(_pair_features(emb[a], emb[b])); y.append(1)
    for a, b in negs:
        X.append(_pair_features(emb[a], emb[b])); y.append(0)
    X, y = np.array(X), np.array(y)

    clf = LogisticRegression(max_iter=2000, C=0.5)
    auc = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
    print(f"pairs: {sum(y)} pos / {len(y)-sum(y)} neg")
    print(f"5-fold ROC-AUC: {auc.mean():.3f} ± {auc.std():.3f}")
    print("(small-data demo of pairwise compatibility ranking; see module docstring)")


if __name__ == "__main__":
    main()
