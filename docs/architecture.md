# Architecture

## One line
A natural-language request is parsed by Gemini, routed to one of two modes, and
answered from **real curated outfits** rather than items assembled from scratch:
FashionCLIP embeds items + outfits into ChromaDB; retrieval + a co-occurrence/
similarity compatibility engine do the ML; Gemini grounds the explanation in the
expert `stylist_rationale`.

## Why outfit-centric (data-driven decision)
The dataset is small and richly labelled: **68 products, 25 expert outfits**,
each with `theme`, `palette`, and a written `stylist_rationale`. It is
**role-structured** (`hero / second / layer / footwear / accessory`), not
topwear/bottomwear, and many outfits are one-piece (dress + footwear + clutch).
Measured facts that shaped the design:

| metric | value |
|---|---|
| items used in outfits | 68 / 68 |
| co-occurrence pairs | 124 |
| items in >1 outfit | 19 / 68 |
| co-occurrence-only recall@5 (LOO) | ~19% |

The 19/68 sparsity is why compatibility **fuses** co-occurrence with FashionCLIP
similarity-transfer instead of relying on co-occurrence alone.

## Data flow

```
OFFLINE (build_index.py)
  products.csv ─► FashionCLIP(image) ⊕ FashionCLIP(text: name+category+tags)
                 ─► fused item vectors ─► ChromaDB[items]
  outfits.csv  ─► FashionCLIP(hero image) ⊕ FashionCLIP(text: theme+palette+occasion)
                 ─► fused outfit vectors ─► ChromaDB[outfits]
  outfits.csv  ─► co-occurrence graph (CompatGraph)

QUERY (recommend.Stylist)
  user message ─► Gemini parse_intent ─► {mode, occasion, gender, age, style, seed}
   │
   ├─ mode = occasion ─► embed query ─► ChromaDB[outfits] (gender filter)
   │                     ─► nearest curated outfit + its items
   │                     ─► Gemini rewrites stylist_rationale for the user (R5)
   │
   └─ mode = seed ─► resolve seed text to nearest catalog item (FashionCLIP)
                     ─► CompatibilityEngine:
                          direct co-occurrence  (W=0.6)
                        + similarity-transfer    (W=0.4)  [borrow partners of
                          visually-similar items]
                        ─► compatible items grouped by complementary type
                     ─► Gemini explanation grounded in palette/formality (R5)
```

## Compatibility engine (Requirement 2)
For a seed item, score every candidate as
`0.6·direct_cooccurrence_norm + 0.4·similarity_transferred_norm`, then group by
the complementary types defined in `config.COMPLEMENTS` and keep the top-N each.
Similarity-transfer = memory-based collaborative filtering with FashionCLIP as
the item-similarity kernel; it generalises the sparse co-occurrence graph.

Optional `compat_model.py` learns a pairwise scorer (logistic regression on
embedding-pair features, positives = co-occurring pairs) — a small-data demo of
the "learning compatibility scores" approach.

## Evaluation (eval.py)
Leave-one-out over the 25 outfits, rebuilding the graph without the held-out
outfit (no leakage), recall@k. Co-occurrence baseline ≈17–19%; pass the
FashionCLIP `similarity_fn` to measure the fused lift.

## Components → evaluation weights
| Component | Choice | Weight served |
|---|---|---|
| Embeddings | FashionCLIP via transformers, image⊕text fusion | CV 15%, ML 30% |
| Retrieval | ChromaDB, dense + gender metadata filter | Retrieval 15% |
| Compatibility | co-occurrence ⊕ similarity-transfer (+ optional learned scorer) | ML 30%, Rec 20% |
| Reasoning | Gemini grounded in stylist_rationale | Rec 20% |
| Eval | leave-one-out recall | rigour / Rec 20% |
| UI + docs | Streamlit, this doc, README | System 10%, Docs 10% |

## Scaling / future work
- Larger catalog → the same pipeline; co-occurrence graph densifies and the
  learned scorer becomes viable.
- Graph neural net (NGNN/HGNN) over items-as-nodes / compatibility-as-edges.
- Per-item colour extraction to add a structured colour-harmony signal.
