"""
Central configuration for the Dare XAI fashion dataset (68 products, 25 outfits).
Edit DATA_DIR to point at your clone of the ML-TASK repo.
"""
from pathlib import Path

# --- paths (EDIT) ----------------------------------------------------------
# CHANGED FROM "data" TO "dataset" TO BYPASS GIT SUBMODULE BUG
DATA_DIR = Path("dataset")              # contains products.csv, outfits.csv, images/
PRODUCTS_CSV = DATA_DIR / "products.csv"
OUTFITS_CSV = DATA_DIR / "outfits.csv"
# products.csv 'image' column is already a path relative to DATA_DIR
CHROMA_DIR = Path("chroma_db")
ITEM_COLLECTION = "items"
OUTFIT_COLLECTION = "outfits"

# --- models ----------------------------------------------------------------
FASHION_CLIP_ID = "patrickjohncyh/fashion-clip"     # via transformers
GEMINI_MODEL = "gemini-2.0-flash"                   # confirm current free model in AI Studio

# --- retrieval / fusion weights --------------------------------------------
TOP_OUTFITS = 3            # nearest curated outfits to retrieve for an occasion query
COMPAT_K_SIMILAR = 5       # k visually-similar items to transfer partners from (seed mode)
COMPAT_TOPN = 4            # compatible items returned per complementary type
W_COOCCUR = 0.6            # weight on direct co-occurrence evidence
W_SIMILAR = 0.4            # weight on similarity-transferred evidence

# --- schema ----------------------------------------------------------------
# (outfit role name, outfit id column) in priority order
ROLE_COLS = [
    ("hero", "hero_id"),
    ("second", "second_id"),
    ("layer", "layer_id"),
    ("footwear", "footwear_id"),
    ("accessory_1", "accessory_1_id"),
    ("accessory_2", "accessory_2_id"),
]

# coarse garment type -> which types complete an outfit around it
COMPLEMENTS = {
    "top":       ["bottom", "footwear", "layer", "accessory"],
    "bottom":    ["top", "footwear", "layer", "accessory"],
    "dress":     ["footwear", "layer", "accessory"],   # one-piece needs no bottom
    "layer":     ["top", "bottom", "footwear", "accessory"],
    "footwear":  ["top", "bottom", "dress", "accessory"],
    "accessory": ["top", "bottom", "dress", "footwear"],
    "apparel":   ["footwear", "accessory", "layer"],
}

# keyword -> coarse type (checked against the 'category' field)
_TYPE_KEYWORDS = {
    "top":    ["shirt", "tshirt", "t-shirt", "top", "sweatshirt", "polo", "blouse", "tee",
               "kurti", "tank", "sweater", "pullover", "activewear"],
    "bottom": ["trouser", "jean", "chino", "pant", "track", "skirt", "palazzo", "legging", "short"],
    "dress":  ["dress", "saree", "gown", "sherwani", "suit", "kurta-set", "sharara",
               "lehenga", "jumpsuit", "set"],
    "layer":  ["blazer", "jacket", "coat", "shrug", "overcoat", "waistcoat"],
}


def coarse_type(category: str, wear_type: str) -> str:
    wt = (wear_type or "").lower()
    if wt == "footwear":
        return "footwear"
    if wt == "accessory":
        return "accessory"
    cat = (category or "").lower()
    for ctype, kws in _TYPE_KEYWORDS.items():
        if any(k in cat for k in kws):
            return ctype
    return "apparel"