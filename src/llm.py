"""
LLM layer (Requirements 4 + 5): Gemini parses intent and writes reasoning that
is GROUNDED in the curated outfit's stylist_rationale (not invented). If
GEMINI_API_KEY is unset or a call fails, deterministic fallbacks keep everything
working -- the stylist_rationale is itself expert text, so explanations stay
high quality offline.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C

try:
    import google.generativeai as genai
except Exception:
    genai = None

_KEY = os.environ.get("GEMINI_API_KEY")
if genai and _KEY:
    genai.configure(api_key=_KEY)


def _model():
    return genai.GenerativeModel(C.GEMINI_MODEL) if (genai and _KEY) else None


def _json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE)
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return json.loads(m.group(0)) if m else {}


# --- intent ----------------------------------------------------------------
_PARSE = """Extract fashion intent as JSON only:
  mode: "seed" ONLY if the user names a specific clothing item they already own
        and want to pair (e.g. "what goes with my white shirt"). If they just
        mention an event/occasion (even "I have a wedding"), mode is "occasion".
  seed_item: the owned item's text if mode=seed, else null
  occasion: e.g. "office","party","wedding","casual","vacation","festive" or ""
  gender: "male","female", or null
  age: integer or null
  style: e.g. "formal","smart casual","ethnic","casual" or ""
User: "{msg}"
JSON:"""


def parse_intent(msg: str) -> dict:
    m = _model()
    if m:
        try:
            d = _json(m.generate_content(_PARSE.format(msg=msg)).text)
            if d:
                return d
        except Exception:
            pass
    return _fallback_intent(msg)


_GARMENT = (r"shirt|t-?shirt|tee|top|blouse|dress|gown|saree|sari|lehenga|kurta|"
            r"kurti|sharara|jeans|trouser|pant|chino|skirt|short|sweater|sweatshirt|"
            r"blazer|jacket|coat|heel|shoe|sneaker|boot|sandal|loafer|jutti|"
            r"watch|bag|clutch|belt|necklace|earring|saree")


def _fallback_intent(msg: str) -> dict:
    t = msg.lower()
    g_re = rf"\b({_GARMENT})s?\b"
    # strong signals that the user wants to style around something they own
    strong = re.search(r"goes with|pair[a-z ]*with|match[a-z ]*with|wear[a-z ]*with|"
                       r"style[a-z ]*(around|with)|build.*around|around (my|them|it|this)|"
                       r"with my", t)
    owns = re.search(r"\bi (have|own|got|bought|wearing)\b", t)
    buying = re.search(r"\b(buy|purchase|shopping|looking to buy|want a|need a|wish to)\b", t)
    seed = bool(strong or (owns and re.search(g_re, t) and not buying))

    occ = next((o for o in ["office", "party", "wedding", "festive", "vacation",
                            "beach", "interview", "sport", "winter", "casual"]
                if o in t), "")
    if occ in ("beach", "vacation"):
        occ = "vacation"
    if occ == "interview":
        occ = "office"
    gender = "male" if re.search(r"\b(male|man|men|guy|boy|sherwani|nehru)\b", t) else \
        "female" if re.search(r"\b(female|woman|women|girl|lehengas?|sarees?|sari|"
                              r"kurtis?|shararas?|dress(es)?|heels?|blouses?)\b", t) else None
    style = "formal" if re.search(r"\b(formal|business|interview|office)\b", t) else \
        "ethnic" if re.search(r"\b(wedding|ethnic|festive|traditional|lehenga|saree|"
                              r"kurta|sharara|sherwani)\b", t) else \
        "smart casual" if "smart casual" in t else "casual"
    age = next((int(n) for n in re.findall(r"\b(\d{2})\b", t)), None)
    return {"mode": "seed" if seed else "occasion",
            "seed_item": msg if seed else None,
            "occasion": occ, "gender": gender, "age": age, "style": style}


# --- explanation (occasion mode) -------------------------------------------
_EXPLAIN_OUTFIT = """You are a fashion stylist. Rewrite the expert note into a
short, friendly explanation (2-3 sentences) tailored to the user's request.
Stay faithful to the expert note; do not invent items.

User asked: "{msg}"
Outfit theme: {theme} | palette: {palette} | occasion: {occasion}
Expert note: {rationale}

Explanation:"""


def explain_outfit(msg: str, outfit: dict) -> str:
    m = _model()
    if m:
        try:
            r = m.generate_content(_EXPLAIN_OUTFIT.format(
                msg=msg, theme=outfit.get("theme", ""),
                palette=outfit.get("palette", ""),
                occasion=outfit.get("occasion", ""),
                rationale=outfit.get("rationale", "")))
            if r.text.strip():
                return r.text.strip()
        except Exception:
            pass
    # fallback: the expert rationale is already good copy
    pal = outfit.get("palette", "")
    return (f"A {outfit.get('theme','')} look in {pal} for {outfit.get('occasion','')}. "
            f"{outfit.get('rationale','')}")


# --- explanation (seed/compatibility mode) ---------------------------------
_EXPLAIN_COMPAT = """You are a fashion stylist. In 2-3 sentences explain why
these pieces pair well with the seed item for the user. Be concrete about
colour/formality. Do not invent items beyond those listed.

Seed item: {seed}
Recommended: {recs}
User context: {ctx}

Explanation:"""


def explain_compat(seed_name: str, recs: dict, ctx: str) -> str:
    # the assembled look = the top pick per type (what the UI shows as the outfit)
    picks = "; ".join(f"{t}: {items[0]['name']}" for t, items in recs.items() if items)
    full = "; ".join(f"{t}: {', '.join(i['name'] for i in items)}"
                     for t, items in recs.items())
    m = _model()
    if m:
        try:
            r = m.generate_content(_EXPLAIN_COMPAT.format(
                seed=seed_name, recs=full, ctx=ctx))
            if r.text.strip():
                return r.text.strip()
        except Exception:
            pass
    return (f"To complete {seed_name}, I paired it with {picks}. These were drawn "
            f"from expert-styled outfits with similar pieces, so they sit at the "
            f"same formality and work as a set.")