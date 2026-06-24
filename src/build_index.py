"""
Precompute step. Embeds the 68 items and 25 outfits with FashionCLIP and stores
them in two persistent ChromaDB collections. Run once (needs internet to pull
the model the first time):

    python src/build_index.py

Multimodal fusion: each vector is the normalised mean of an image embedding and
a text embedding (same CLIP space), so text queries match well and visual
similarity still works for the compatibility transfer.
"""
import sys
from pathlib import Path

import chromadb
import numpy as np
import pandas as pd
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as C
from data import load_outfits, load_products, outfit_item_ids
from embeddings import FashionEncoder


def _fuse(img_vec: np.ndarray, txt_vec: np.ndarray) -> np.ndarray:
    v = (img_vec + txt_vec) / 2.0
    return v / (np.linalg.norm(v) + 1e-9)


def _open(path: str):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def build_items(enc, prods, coll):
    imgs, texts, keep = [], [], []
    for r in prods.to_dict("records"):
        im = _open(r["image_path"])
        if im is None:
            print("  skip (no image):", r["id"])
            continue
        imgs.append(im)
        texts.append(f"{r['name']}. {r['category']}. {r.get('tags','')}")
        keep.append(r)
    iv = enc.embed_images(imgs)
    tv = enc.embed_texts(texts)
    vecs = [_fuse(iv[i], tv[i]).tolist() for i in range(len(keep))]
    coll.add(
        ids=[r["id"] for r in keep],
        embeddings=vecs,
        metadatas=[{
            "name": r["name"], "category": r["category"],
            "coarse_type": r["coarse_type"], "gender": str(r["gender"]),
            "wear_type": str(r["wear_type"]), "occasion": str(r["occasion"]),
            "image_path": r["image_path"],
        } for r in keep],
    )
    print(f"items indexed: {coll.count()}")


def build_outfits(enc, outs, prod_lookup, coll):
    imgs, texts, metas, ids = [], [], [], []
    for r in outs.to_dict("records"):
        hero = prod_lookup.get(r["hero_id"])
        im = _open(hero["image_path"]) if hero else None
        if im is None:
            print("  skip outfit (no hero image):", r["outfit_id"])
            continue
        imgs.append(im)
        texts.append(f"{r['theme']}. {r['palette']}. {r['occasion']} "
                     f"{r['wear_type']} for {r['gender']}.")
        item_ids = outfit_item_ids(r)
        metas.append({
            "outfit_id": r["outfit_id"], "gender": str(r["gender"]),
            "occasion": str(r["occasion"]), "wear_type": str(r["wear_type"]),
            "theme": str(r["theme"]), "palette": str(r["palette"]),
            "rationale": str(r["stylist_rationale"]),
            "item_ids": ",".join(item_ids),
        })
        ids.append(r["outfit_id"])
    iv = enc.embed_images(imgs)
    tv = enc.embed_texts(texts)
    vecs = [_fuse(iv[i], tv[i]).tolist() for i in range(len(ids))]
    coll.add(ids=ids, embeddings=vecs, metadatas=metas)
    print(f"outfits indexed: {coll.count()}")


def main():
    prods, outs = load_products(), load_outfits()
    lookup = {r["id"]: r for r in prods.to_dict("records")}
    enc = FashionEncoder(C.FASHION_CLIP_ID)
    client = chromadb.PersistentClient(path=str(C.CHROMA_DIR))
    for name in (C.ITEM_COLLECTION, C.OUTFIT_COLLECTION):
        try:
            client.delete_collection(name)
        except Exception:
            pass
    items = client.create_collection(C.ITEM_COLLECTION, metadata={"hnsw:space": "cosine"})
    outfits = client.create_collection(C.OUTFIT_COLLECTION, metadata={"hnsw:space": "cosine"})
    build_items(enc, prods, items)
    build_outfits(enc, outs, lookup, outfits)
    print("done ->", C.CHROMA_DIR)


if __name__ == "__main__":
    main()
