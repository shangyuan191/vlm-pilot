"""Compact index of the Amazon Reviews 2023 complete-field subsets (furniture, decor, lighting) for pool building.

Reads data/external/amazon_reviews_2023/{home_decor,lighting}_complete.jsonl.gz once and writes
data/derived/amazon_index.jsonl.gz with: asin, title, leaf/path categories, style/color/material/shape from
`details`, price, rating_number, store, first large image URL.  No network access.
"""
import ast, gzip, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/external/amazon_reviews_2023"
OUT = ROOT / "data/derived/amazon_index.jsonl.gz"
KEEP = ("Furniture", "Home Décor Products", "Lighting & Ceiling Fans", "Kitchen & Dining")
DETAIL_KEYS = {"Style": "style", "Color": "color", "Material": "material", "Shape": "shape", "Finish Type": "finish", "Product Dimensions": "dims"}


def lit(s, default):
    try: return ast.literal_eval(s) if isinstance(s, str) else (s if s is not None else default)
    except Exception: return default


n_in = n_out = 0
with gzip.open(OUT, "wt") as out:
    for f in ("home_decor_complete.jsonl.gz", "lighting_complete.jsonl.gz"):
        for line in gzip.open(SRC / f, "rt"):
            n_in += 1
            r = json.loads(line)
            cats = lit(r.get("categories"), [])
            if not any(k in cats for k in KEEP): continue
            det = lit(r.get("details"), {}) or {}
            imgs = lit(r.get("images"), []) or []
            img = next((i.get("large") or i.get("hi_res") for i in imgs if isinstance(i, dict) and (i.get("large") or i.get("hi_res"))), None)
            if not img: continue
            rec = {"asin": r["parent_asin"], "title": r.get("title"), "cats": cats[1:], "price": r.get("price"),
                   "rating_number": r.get("rating_number"), "store": r.get("store"), "img": img, "src": f.split("_complete")[0]}
            for k, v in DETAIL_KEYS.items():
                if det.get(k): rec[v] = str(det[k])
            out.write(json.dumps(rec, ensure_ascii=False) + "\n"); n_out += 1
print(n_in, "read,", n_out, "kept ->", OUT)
