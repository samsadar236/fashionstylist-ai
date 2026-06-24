"""
Conversational fashion assistant UI (Requirement 4).

    streamlit run src/app.py

Build the index first: python src/build_index.py

Design: warm editorial "lookbook". Left = conversation, right = the styled
outfit canvas with palette, cohesion, and price. Profile + occasion chips in the
sidebar.
"""
import base64
import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))
from recommend import Stylist

st.set_page_config(page_title="AURA · AI Stylist", page_icon="🧵", layout="wide")

# ---------------------------------------------------------------- styling ----
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600&display=swap');
:root{
 --bg:#EFE7DA; --panel:#F6F0E6; --card:#FCFAF5; --ink:#2E2A24; --muted:#9A8E7D;
 --line:#E3D8C6; --clay:#B6533B; --sage:#6E7B5B;
}
[data-testid="stAppViewContainer"]{background:var(--bg);}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stSidebar"]{background:var(--panel);border-right:1px solid var(--line);}
[data-testid="stSidebar"] *{color:var(--ink);}
.block-container{padding-top:1.6rem;max-width:1280px;}
html,body,[class*="css"],p,div,span,label{font-family:'Inter',system-ui,sans-serif;color:var(--ink);}
h1,h2,h3{font-family:'Fraunces',Georgia,serif;font-weight:600;letter-spacing:-.01em;}
.app-title{font-family:'Fraunces',serif;font-size:2.4rem;font-weight:700;margin:0;letter-spacing:-.02em;}
.app-sub{color:var(--muted);font-size:.92rem;margin:.1rem 0 1.1rem;}
/* sidebar buttons = occasion chips */
[data-testid="stSidebar"] .stButton>button{
 background:var(--card);border:1px solid var(--line);border-radius:999px;
 color:var(--ink);font-weight:500;font-size:.85rem;padding:.32rem .7rem;width:100%;}
[data-testid="stSidebar"] .stButton>button:hover{border-color:var(--clay);color:var(--clay);}
/* chat bubbles (left) */
.bubble{border-radius:14px;padding:.7rem .9rem;margin:.4rem 0;font-size:.92rem;line-height:1.45;}
.bubble.you{background:var(--clay);color:#FBF5EF;border-bottom-right-radius:4px;}
.bubble.aura{background:var(--card);border:1px solid var(--line);border-bottom-left-radius:4px;}
.bubble .who{font-size:.68rem;letter-spacing:.08em;text-transform:uppercase;opacity:.7;margin-bottom:.2rem;}
/* lookbook canvas (right) */
.lookbook{background:var(--card);border:1px solid var(--line);border-radius:18px;
 padding:1.3rem 1.4rem 1.5rem;box-shadow:0 18px 40px -28px rgba(46,42,36,.5);}
.lb-eyebrow{font-size:.7rem;letter-spacing:.16em;text-transform:uppercase;color:var(--clay);font-weight:600;}
.lb-title{font-family:'Fraunces',serif;font-size:1.9rem;font-weight:600;margin:.15rem 0 .5rem;line-height:1.1;}
.lb-stats{display:flex;gap:1.1rem;align-items:center;flex-wrap:wrap;margin-bottom:.5rem;}
.stat{font-size:.82rem;color:var(--muted);}.stat b{color:var(--ink);font-size:1rem;font-family:'Fraunces',serif;}
.stat.total{margin-left:auto;font-family:'Fraunces',serif;font-size:1.25rem;color:var(--ink);}
.meter{height:5px;background:var(--line);border-radius:99px;overflow:hidden;margin:.1rem 0 .9rem;}
.meter>i{display:block;height:100%;background:linear-gradient(90deg,var(--sage),var(--clay));}
.palette{display:flex;gap:.35rem;margin:.2rem 0 1rem;}
.sw{width:26px;height:26px;border-radius:7px;border:1px solid rgba(0,0,0,.08);}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:.8rem;}
.item{background:var(--bg);border:1px solid var(--line);border-radius:13px;overflow:hidden;}
.item.seed{outline:2px solid var(--clay);outline-offset:-2px;}
.thumb{width:100%;aspect-ratio:3/4;background-size:cover;background-position:center;background-color:#E7DCCB;}
.it-meta{padding:.45rem .55rem .6rem;}
.it-type{font-size:.64rem;letter-spacing:.1em;text-transform:uppercase;color:var(--sage);font-weight:600;}
.it-name{font-size:.78rem;line-height:1.25;margin:.12rem 0 .25rem;min-height:2rem;}
.it-price{font-family:'Fraunces',serif;font-size:.92rem;}
.alts{margin-top:1rem;border-top:1px solid var(--line);padding-top:.8rem;}
.alts h4{font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin:.2rem 0 .5rem;font-family:'Inter',sans-serif;font-weight:600;}
.tag{display:inline-block;background:var(--bg);border:1px solid var(--line);border-radius:999px;
 padding:.2rem .6rem;font-size:.72rem;margin:0 .3rem .3rem 0;color:var(--muted);}
.empty{color:var(--muted);font-size:.95rem;border:1px dashed var(--line);border-radius:16px;padding:2rem;text-align:center;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------- helpers ----
_HEX = {
 "black": "#2E2A24", "white": "#FBF6EC", "cream": "#EFE3CF", "off white": "#F2EAD9",
 "beige": "#D9C7AA", "tan": "#C9A981", "khaki": "#B6A268", "nude": "#E2C9AE",
 "brown": "#7A5230", "coffee": "#5C3D2E", "coffee brown": "#5C3D2E", "rust": "#A0522D",
 "maroon": "#6E2433", "burgundy": "#5E2031", "red": "#B23A2E", "pink": "#D98C9A",
 "rose": "#C56E78", "peach": "#E7B89B", "mauve": "#B08DA0", "orange": "#C8702D",
 "mustard": "#C99A2E", "gold": "#C9A227", "yellow": "#E3C04B", "olive": "#6E7B3D",
 "green": "#5C7148", "sage": "#8A9A6F", "lime green": "#8FA63B", "teal": "#2E6E6A",
 "blue": "#3E5C86", "navy": "#2C3A57", "purple": "#5E4B7B", "lavender": "#9A8FB5",
 "grey": "#9A8E7D", "gray": "#9A8E7D", "charcoal": "#3D3A34", "silver": "#BFB6A8",
}


@st.cache_data(show_spinner=False)
def b64(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode()


def swatches(palette: str) -> str:
    out = []
    for tok in str(palette).split("/"):
        t = tok.strip().lower()
        out.append(f'<span class="sw" style="background:{_HEX.get(t, "#C9BBA6")}" title="{tok.strip()}"></span>')
    return "".join(out)


def money(v) -> str:
    return f"₹{v:,}" if isinstance(v, int) else "—"


def item_card(it: dict, seed: bool = False) -> str:
    img = b64(it.get("image_path", ""))
    bg = f"background-image:url('{img}')" if img else ""
    return (f'<div class="item{" seed" if seed else ""}">'
            f'<div class="thumb" style="{bg}"></div>'
            f'<div class="it-meta"><div class="it-type">{it.get("type","")}'
            f'{" · your item" if seed else ""}</div>'
            f'<div class="it-name">{it.get("name","")}</div>'
            f'<div class="it-price">{money(it.get("price"))}</div></div></div>')


def pct(v):
    return round(v * 100) if isinstance(v, (int, float)) else None


def render_canvas(rec: dict) -> str:
    if rec["mode"] == "occasion":
        o = rec.get("outfit")
        if not o:
            return f'<div class="empty">{rec["explanation"]}</div>'
        coh = pct(rec.get("cohesion"))
        stats = (f'<span class="stat"><b>{pct(o["match"])}%</b> match</span>'
                 f'<span class="stat"><b>{coh}%</b> cohesion</span>'
                 f'<span class="stat total">{money(rec.get("total_price"))}</span>')
        meter = f'<div class="meter"><i style="width:{coh or 0}%"></i></div>'
        cards = "".join(item_card(i) for i in rec["items"])
        return (f'<div class="lookbook"><div class="lb-eyebrow">your look · {o["occasion"]}</div>'
                f'<div class="lb-title">{o["theme"]}</div><div class="lb-stats">{stats}</div>'
                f'{meter}<div class="palette">{swatches(o["palette"])}</div>'
                f'<div class="grid">{cards}</div></div>')

    # seed mode
    s = rec["seed"]
    coh = pct(rec.get("cohesion"))
    picks = [g[0] for g in rec["groups"].values() if g]
    stats = (f'<span class="stat"><b>{coh}%</b> cohesion</span>'
             f'<span class="stat total">{money(rec.get("total_price"))}</span>')
    meter = f'<div class="meter"><i style="width:{coh or 0}%"></i></div>'
    cards = item_card(s, seed=True) + "".join(item_card(p) for p in picks)
    alts = ""
    extra = {t: g[1:] for t, g in rec["groups"].items() if len(g) > 1}
    if extra:
        rows = ""
        for t, items in extra.items():
            tags = "".join(f'<span class="tag">{i["name"]} · {money(i["price"])}</span>'
                           for i in items)
            rows += f'<div><div class="it-type">{t}</div>{tags}</div>'
        alts = f'<div class="alts"><h4>Other options</h4>{rows}</div>'
    return (f'<div class="lookbook"><div class="lb-eyebrow">styled around your {s.get("type","item")}</div>'
            f'<div class="lb-title">{s["name"]}</div><div class="lb-stats">{stats}</div>'
            f'{meter}<div class="grid">{cards}</div>{alts}</div>')


# ---------------------------------------------------------------- backend ----
@st.cache_resource(show_spinner="Loading FashionCLIP + index…")
def get_stylist() -> Stylist:
    return Stylist()


# ---------------------------------------------------------------- sidebar ----
st.sidebar.markdown("### Your profile")
g = st.sidebar.selectbox("Who's it for", ["Anyone", "Women", "Men"], key="pf_g")
style = st.sidebar.selectbox("Style", ["Any", "Formal", "Smart casual", "Casual",
                                       "Ethnic", "Party"], key="pf_s")
age = st.sidebar.slider("Age", 16, 70, 25, key="pf_a")
profile = {"gender": None if g == "Anyone" else g.lower(),
           "style": None if style == "Any" else style.lower(), "age": age}

st.sidebar.markdown("### Quick occasions")
chips = ["Wedding", "Office", "Party", "Casual", "Festive", "Vacation"]
cc = st.sidebar.columns(2)
for n, label in enumerate(chips):
    if cc[n % 2].button(label, key=f"chip_{label}"):
        st.session_state.pending = f"an outfit for a {label.lower()}"
st.sidebar.markdown("<br>", unsafe_allow_html=True)
if st.sidebar.button("Clear conversation", key="clear"):
    st.session_state.turns = []

# ---------------------------------------------------------------- main -------
st.markdown('<div class="app-title">AURA</div>'
            '<div class="app-sub">Tell me an occasion, or an item you own, and I\'ll style the look.</div>',
            unsafe_allow_html=True)

if "turns" not in st.session_state:
    st.session_state.turns = []

left, right = st.columns([5, 6], gap="large")

with left:
    if not st.session_state.turns:
        st.markdown('<div class="bubble aura"><div class="who">Aura</div>'
                    'Try “smart casual for an office day” or “what goes with my black heels?”'
                    '</div>', unsafe_allow_html=True)
    for t in st.session_state.turns:
        st.markdown(f'<div class="bubble you"><div class="who">You</div>{t["msg"]}</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="bubble aura"><div class="who">Aura</div>{t["rec"]["explanation"]}</div>',
                    unsafe_allow_html=True)

with right:
    if st.session_state.turns:
        st.markdown(render_canvas(st.session_state.turns[-1]["rec"]),
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty">Your styled outfit will appear here.</div>',
                    unsafe_allow_html=True)

# input (full width, pinned bottom) + chip handling
prompt = st.chat_input("Describe an occasion or an item…")
if st.session_state.get("pending"):
    prompt = st.session_state.pop("pending")
if prompt:
    with st.spinner("Styling…"):
        rec = get_stylist().recommend(prompt, profile)
    st.session_state.turns.append({"msg": prompt, "rec": rec})
    st.rerun()
