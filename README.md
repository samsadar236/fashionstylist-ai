# AI Fashion Outfit Recommendation System — Dare XAI

A conversational stylist over a curated catalog (**68 products, 25 expert
outfits**). Two modes:
- **Occasion** — "outfit for a wedding" → retrieves the nearest curated outfit
  and explains it, grounded in the expert `stylist_rationale`.
- **Seed item** — "what goes with my white formal shirt?" → a compatibility
  engine returns matching pieces by complementary type.

FashionCLIP (image+text) for embeddings, ChromaDB for retrieval, a
co-occurrence + similarity-transfer compatibility engine for the ML, Gemini for
intent parsing and reasoning.

## Requirement → code map
| Req | Where |
|---|---|
| 1 Dataset analysis | `src/inspect_dataset.py` → `docs/dataset_analysis.md` |
| 2 Compatibility engine | `src/compatibility.py` (+ optional `src/compat_model.py`) |
| 3 User-aware | gender filter + intent in `retriever.py`, `llm.py` |
| 4 Conversational | `src/app.py`, `llm.parse_intent` |
| 5 Explainability | `llm.explain_outfit` / `explain_compat` (grounded in rationale) |
| Eval (rigour) | `src/eval.py` — leave-one-out recall |

## Setup
```bash
pip install -r requirements.txt
git clone https://github.com/DarexAI-AI-Startup/ML-TASK.git data   # -> data/products.csv, data/outfits.csv, data/images/
# (PowerShell) Gemini key — optional; system runs without it via fallbacks
$env:GEMINI_API_KEY = "your_key"
```
If `data/` differs, set `DATA_DIR` in `config.py`.

## Run (in order)
```bash
python src/inspect_dataset.py                       # Req 1 (no model needed)
python src/eval.py                                  # co-occurrence baseline (no model)
python src/build_index.py                           # embed items + outfits -> ChromaDB (one-time)
python src/recommend.py "smart casual for a dinner date, 24M"
python src/recommend.py "I have a white formal shirt, what goes with it?"
python src/compat_model.py                          # optional ML-depth experiment
streamlit run src/app.py                            # demo UI
```

## Key design decisions (for the video / interview)
- **Outfit-centric, not slot-centric.** The data is role-structured with 25
  labelled outfits + rationales; recommending from proven combinations beats
  assembling top+bottom+footwear from scratch, and avoids the dress/one-piece
  failure case.
- **Compatibility fuses two signals because co-occurrence is sparse.** Measured:
  only 19/68 items repeat across outfits → co-occurrence-only recall@5 ≈19%. So
  we add FashionCLIP similarity-transfer (borrow partners of visually similar
  items). Empirically motivated, not asserted.
- **Explanations are grounded.** Gemini rewrites the expert `stylist_rationale`
  for the user instead of inventing reasoning — and the system still works with
  no API key because the rationale is already expert text.
- **FashionCLIP over vanilla CLIP**: fashion-fine-tuned, separates
  style/formality/colour far better on this domain, zero training.

## Honesty notes
- The compatibility numbers (recall@k) are real and modest — the dataset is tiny
  by design; the PS values approach over accuracy.
- `compat_model.py` is a small-data demonstration of pairwise ranking, not a
  production model (124 positive pairs).
- No per-item colour field exists; colour reasoning comes from outfit palette +
  LLM, not a structured column.

## Layout
```
config.py                 paths, schema, fusion weights, type mapping
src/data.py               loaders + image paths + co-occurrence graph
src/inspect_dataset.py    Req 1 analysis
src/embeddings.py         FashionCLIP wrapper (transformers)
src/build_index.py        embed items + outfits -> ChromaDB
src/compatibility.py      Req 2: co-occurrence + similarity-transfer
src/compat_model.py       optional learned pairwise scorer
src/retriever.py          outfit retrieval + item similarity_fn
src/llm.py                Req 4/5: intent + grounded reasoning
src/recommend.py          orchestrator (occasion + seed modes)
src/eval.py               leave-one-out recall
src/app.py                Streamlit chat UI
docs/architecture.md      design + rationale
```
