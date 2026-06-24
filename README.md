# AURA — AI Fashion Outfit Recommendation System
**Dare XAI · ML & AI Engineer Intern Assignment**

A conversational fashion stylist. Describe an occasion or an item you own — AURA returns a complete, compatible outfit with a reason for every pick. Built on FashionCLIP embeddings, ChromaDB vector retrieval, a co-occurrence + similarity-transfer compatibility engine, and Gemini for intent parsing and grounded explanations.

---

## System Architecture

<p align="center">
<svg width="680" viewBox="0 0 680 860" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;font-family:system-ui,sans-serif">
<defs>
<marker id="ar" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M2 1L8 5L2 9" fill="none" stroke="#888" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</marker>
</defs>

<!-- offline zone -->
<rect x="40" y="42" width="600" height="248" rx="12" fill="none" stroke="#ccc" stroke-width="1" stroke-dasharray="5 4"/>
<text x="56" y="60" font-size="11" fill="#999" font-weight="500">OFFLINE — build once (build_index.py)</text>

<!-- data sources -->
<rect x="60" y="70" width="160" height="56" rx="8" fill="#E1F5EE" stroke="#0F6E56" stroke-width="0.8"/>
<text x="140" y="93" font-size="13" font-weight="600" fill="#085041" text-anchor="middle">products.csv</text>
<text x="140" y="112" font-size="11" fill="#0F6E56" text-anchor="middle">68 items + image paths</text>

<rect x="250" y="70" width="170" height="56" rx="8" fill="#E1F5EE" stroke="#0F6E56" stroke-width="0.8"/>
<text x="335" y="93" font-size="13" font-weight="600" fill="#085041" text-anchor="middle">outfits.csv</text>
<text x="335" y="112" font-size="11" fill="#0F6E56" text-anchor="middle">25 outfits + rationale</text>

<rect x="450" y="70" width="170" height="56" rx="8" fill="#E1F5EE" stroke="#0F6E56" stroke-width="0.8"/>
<text x="535" y="93" font-size="13" font-weight="600" fill="#085041" text-anchor="middle">CompatGraph</text>
<text x="535" y="112" font-size="11" fill="#0F6E56" text-anchor="middle">124 co-occur pairs</text>

<line x1="420" y1="98" x2="450" y2="98" stroke="#888" stroke-width="1" marker-end="url(#ar)"/>

<!-- FashionCLIP -->
<rect x="110" y="168" width="320" height="56" rx="8" fill="#EEEDFE" stroke="#534AB7" stroke-width="0.8"/>
<text x="270" y="191" font-size="13" font-weight="600" fill="#26215C" text-anchor="middle">FashionCLIP encoder</text>
<text x="270" y="210" font-size="11" fill="#534AB7" text-anchor="middle">ViT-B/32 · image ⊕ text → 512-d vector</text>

<line x1="140" y1="126" x2="190" y2="168" stroke="#888" stroke-width="1" marker-end="url(#ar)"/>
<line x1="335" y1="126" x2="310" y2="168" stroke="#888" stroke-width="1" marker-end="url(#ar)"/>

<!-- Chroma stores -->
<rect x="60" y="262" width="170" height="56" rx="8" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="0.8"/>
<text x="145" y="285" font-size="13" font-weight="600" fill="#2C2C2A" text-anchor="middle">Chroma · items</text>
<text x="145" y="304" font-size="11" fill="#5F5E5A" text-anchor="middle">68 item vectors</text>

<rect x="250" y="262" width="180" height="56" rx="8" fill="#F1EFE8" stroke="#5F5E5A" stroke-width="0.8"/>
<text x="340" y="285" font-size="13" font-weight="600" fill="#2C2C2A" text-anchor="middle">Chroma · outfits</text>
<text x="340" y="304" font-size="11" fill="#5F5E5A" text-anchor="middle">25 outfit vectors</text>

<line x1="200" y1="224" x2="175" y2="262" stroke="#888" stroke-width="1" marker-end="url(#ar)"/>
<line x1="310" y1="224" x2="320" y2="262" stroke="#888" stroke-width="1" marker-end="url(#ar)"/>

<!-- query zone -->
<rect x="40" y="340" width="600" height="480" rx="12" fill="none" stroke="#ccc" stroke-width="1" stroke-dasharray="5 4"/>
<text x="56" y="358" font-size="11" fill="#999" font-weight="500">QUERY PIPELINE — every request</text>

<!-- user message -->
<rect x="215" y="368" width="250" height="44" rx="8" fill="#EEEDFE" stroke="#534AB7" stroke-width="0.8"/>
<text x="340" y="384" font-size="13" font-weight="600" fill="#26215C" text-anchor="middle">User message</text>
<text x="340" y="402" font-size="11" fill="#534AB7" text-anchor="middle">"outfit for a wedding" / "I have heels…"</text>

<!-- parse_intent -->
<rect x="180" y="444" width="320" height="56" rx="8" fill="#EEEDFE" stroke="#534AB7" stroke-width="0.8"/>
<text x="340" y="465" font-size="13" font-weight="600" fill="#26215C" text-anchor="middle">parse_intent</text>
<text x="340" y="484" font-size="11" fill="#534AB7" text-anchor="middle">Gemini + keyword fallback · mode / gender / occasion</text>

<line x1="340" y1="412" x2="340" y2="444" stroke="#888" stroke-width="1" marker-end="url(#ar)"/>
<text x="340" y="524" font-size="11" fill="#aaa" text-anchor="middle">mode branch</text>
<text x="148" y="546" font-size="11" fill="#993C1D" text-anchor="middle" font-weight="500">occasion mode</text>
<text x="532" y="546" font-size="11" fill="#854F0B" text-anchor="middle" font-weight="500">seed mode</text>

<!-- occasion branch -->
<rect x="50" y="556" width="196" height="56" rx="8" fill="#FAECE7" stroke="#993C1D" stroke-width="0.8"/>
<text x="148" y="577" font-size="13" font-weight="600" fill="#4A1B0C" text-anchor="middle">Outfit retrieval</text>
<text x="148" y="596" font-size="11" fill="#993C1D" text-anchor="middle">Chroma outfits · gender filter</text>

<rect x="50" y="648" width="196" height="56" rx="8" fill="#FAECE7" stroke="#993C1D" stroke-width="0.8"/>
<text x="148" y="669" font-size="13" font-weight="600" fill="#4A1B0C" text-anchor="middle">explain_outfit</text>
<text x="148" y="688" font-size="11" fill="#993C1D" text-anchor="middle">Gemini rewrites rationale</text>

<!-- seed branch -->
<rect x="434" y="556" width="196" height="56" rx="8" fill="#FAEEDA" stroke="#854F0B" stroke-width="0.8"/>
<text x="532" y="577" font-size="13" font-weight="600" fill="#412402" text-anchor="middle">CompatibilityEngine</text>
<text x="532" y="596" font-size="11" fill="#854F0B" text-anchor="middle">co-occur ×0.6 + CLIP sim ×0.4</text>

<rect x="434" y="648" width="196" height="56" rx="8" fill="#FAEEDA" stroke="#854F0B" stroke-width="0.8"/>
<text x="532" y="669" font-size="13" font-weight="600" fill="#412402" text-anchor="middle">explain_compat</text>
<text x="532" y="688" font-size="11" fill="#854F0B" text-anchor="middle">Gemini: colour + formality</text>

<!-- branch arrows -->
<path d="M280 500 L280 530 L148 530 L148 556" fill="none" stroke="#888" stroke-width="1" marker-end="url(#ar)"/>
<path d="M400 500 L400 530 L532 530 L532 556" fill="none" stroke="#888" stroke-width="1" marker-end="url(#ar)"/>
<line x1="148" y1="612" x2="148" y2="648" stroke="#888" stroke-width="1" marker-end="url(#ar)"/>
<line x1="532" y1="612" x2="532" y2="648" stroke="#888" stroke-width="1" marker-end="url(#ar)"/>

<!-- Streamlit output -->
<rect x="165" y="742" width="350" height="44" rx="8" fill="#EEEDFE" stroke="#534AB7" stroke-width="0.8"/>
<text x="340" y="758" font-size="13" font-weight="600" fill="#26215C" text-anchor="middle">Streamlit UI</text>
<text x="340" y="776" font-size="11" fill="#534AB7" text-anchor="middle">lookbook cards · palette · cohesion · price</text>

<path d="M148 704 L148 764 L165 764" fill="none" stroke="#888" stroke-width="1" marker-end="url(#ar)"/>
<path d="M532 704 L532 764 L515 764" fill="none" stroke="#888" stroke-width="1" marker-end="url(#ar)"/>

<!-- cross-tier dashed -->
<path d="M340 318 L340 340 L340 340 L148 340 L148 556" fill="none" stroke="#aaa" stroke-width="0.8" stroke-dasharray="4 3" marker-end="url(#ar)"/>
<path d="M620 98 L640 98 L640 584 L630 584" fill="none" stroke="#aaa" stroke-width="0.8" stroke-dasharray="4 3" marker-end="url(#ar)"/>
<path d="M145 318 L145 820 L532 820 L532 704" fill="none" stroke="#aaa" stroke-width="0.8" stroke-dasharray="4 3" marker-end="url(#ar)"/>

<!-- legend -->
<rect x="40" y="826" width="10" height="10" rx="2" fill="#E1F5EE" stroke="#0F6E56" stroke-width="0.8"/>
<text x="56" y="836" font-size="11" fill="#555">Data / storage</text>
<rect x="160" y="826" width="10" height="10" rx="2" fill="#EEEDFE" stroke="#534AB7" stroke-width="0.8"/>
<text x="176" y="836" font-size="11" fill="#555">ML model / user</text>
<rect x="300" y="826" width="10" height="10" rx="2" fill="#FAECE7" stroke="#993C1D" stroke-width="0.8"/>
<text x="316" y="836" font-size="11" fill="#555">Occasion mode</text>
<rect x="430" y="826" width="10" height="10" rx="2" fill="#FAEEDA" stroke="#854F0B" stroke-width="0.8"/>
<text x="446" y="836" font-size="11" fill="#555">Seed mode</text>
</svg>
</p>

---

## Requirements → code

| # | Requirement | File |
|---|---|---|
| 1 | Dataset analysis | `src/inspect_dataset.py` → `docs/dataset_analysis.md` |
| 2 | Compatibility engine | `src/compatibility.py` + `src/data.py` |
| 3 | User-aware recommendations | `src/llm.py` (intent) + `src/app.py` (profile sidebar) |
| 4 | Conversational assistant | `src/app.py` (Streamlit chat) |
| 5 | Explainability | `src/llm.py` (Gemini rewrites expert rationale) |

---

## Technical approach

### FashionCLIP — Computer Vision
Model: `patrickjohncyh/fashion-clip` (ViT-B/32, fashion fine-tuned). Each item embedded as the normalised mean of image + text vectors (512-d). Joint embedding space means text queries match image-derived vectors out of the box.

### Hybrid compatibility engine
Two signals are fused because the data is sparse: only 19/68 items appear in more than one outfit, making co-occurrence alone weak (leave-one-out recall@5 ≈ 19%).

- **Co-occurrence** (weight 0.6) — which items actually appear together in the 25 curated outfits
- **Similarity-transfer** (weight 0.4) — borrow partners from FashionCLIP-similar items (memory-based CF with a visual kernel)

### ChromaDB retrieval
Two persistent collections: `items` (68 vectors) and `outfits` (25 vectors). Outfit retrieval uses dense ANN + metadata `where` filter on gender. Item similarity uses in-memory matrix multiply across the 68-item matrix.

### RAG pipeline
Nearest curated outfit → `theme` + `palette` + `stylist_rationale` passed to Gemini → response rewritten for the user's specific query. Grounded, not invented. System works fully without a Gemini key (rationale used directly as fallback).

### Evaluation
| Method | Recall@3 | Recall@5 |
|---|---|---|
| Co-occurrence only (baseline) | 17% | 19% |
| + FashionCLIP similarity-transfer | run `python src/eval.py` | — |

---

## Setup

```bash
git clone https://github.com/samsadar236/fashionstylist-ai.git
cd fashionstylist-ai
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows
pip install -r requirements.txt
git clone https://github.com/DarexAI-AI-Startup/ML-TASK.git data
# Optional Gemini key (system runs without it)
$env:GEMINI_API_KEY = "your_key"
```

## Run

```bash
python src/inspect_dataset.py   # dataset analysis — no model needed
python src/eval.py              # co-occurrence baseline
python src/build_index.py       # embed 68 items → ChromaDB (one-time, ~2 min)
python src/recommend.py "smart casual for a dinner date, 24M"
python src/recommend.py "what goes with my white formal shirt?"
python src/compat_model.py      # optional: learned pairwise scorer
streamlit run src/app.py        # demo UI
```

> Run all commands from the project root (where `config.py` lives).

---

## File structure

```
config.py                  Paths, model IDs, fusion weights
src/
  data.py                  CSV loaders, image resolver, co-occurrence graph
  embeddings.py            FashionCLIP wrapper (transformers)
  build_index.py           Embed items + outfits → ChromaDB (run once)
  inspect_dataset.py       Req 1: dataset analysis
  compatibility.py         Req 2: co-occur + similarity-transfer engine
  compat_model.py          Optional: learned pairwise compatibility scorer
  retriever.py             Outfit ANN + item similarity_fn + cohesion score
  llm.py                   Intent parsing + grounded explanations
  recommend.py             Orchestrator: occasion + seed modes
  eval.py                  Leave-one-out recall evaluation
  app.py                   Streamlit conversational UI
docs/
  architecture.md          Detailed design rationale
  dataset_analysis.md      Generated by inspect_dataset.py
```

---

## Design decisions

**Outfit-centric.** The dataset provides 25 expert outfits with rationales — recommending from proven combinations and grounding explanations in real stylist copy beats assembling pieces from scratch.

**Fusion justified by measurement.** Co-occurrence alone hits 17% recall@3. The embedding similarity-transfer path generalises across the sparsity gap.

**Graceful degradation.** No Gemini key → stylist rationale used directly (already expert copy). No internet → index is pre-built. Full functionality in both scenarios.

**Honest limitations.** 68-item catalog: the system returns the closest real item, never invents one. On-model lifestyle images add background noise to embeddings (future: garment segmentation). Co-occurrence graph is sparse by design.

---

## Future improvements
- Garment segmentation before embedding (remove background noise from lifestyle shots)
- Dense + BM25 sparse hybrid retrieval
- Trained compatibility head (OutfitTransformer-style on Polyvore)
- Graph neural network (NGNN/HGNN) over the co-occurrence graph
