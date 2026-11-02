"""Pilot 1: export the candidate pools (top-10 per engine, same as vlm_subset_selection.py) for 20 queries."""
import json, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
QUERY_IDS = [1, 8, 11, 13, 26, 28, 30, 33, 38, 41, 43, 44, 49, 50, 71, 72, 74, 75, 76, 86]
ENGINES = ["amazon", "google_shopping", "onton"]

rows = {int(json.loads(l)["query_id"]): json.loads(l) for l in open(ROOT / "parsed/queries.jsonl")}
tax = {int(r["query_id"]): r for r in json.load(open(ROOT / "query_taxonomy.json"))["rows"]}
img_dir = OUT / "web/img"
img_dir.mkdir(parents=True, exist_ok=True)

pool = []
for q in QUERY_IDS:
    r = rows[q]
    entry = {"query_id": q, "query": r["query"], "tags": tax.get(q, {}).get("tags", []), "items": []}
    for e in ENGINES:
        for pos, it in enumerate(r["engines"][e].get("items", [])[:10], start=1):
            crop = it.get("crop_path")
            img = None
            if crop and (ROOT / "parsed" / crop).exists():
                img = f"img/{q}_{e[0]}_{pos:02d}.png"
                shutil.copy(ROOT / "parsed" / crop, OUT / "web" / img)
            entry["items"].append({
                "item_id": it["item_id"], "engine": e, "position": pos,
                "title": it.get("title"), "price": it.get("price"), "seller": it.get("seller"),
                "title_truncated": it.get("title_truncated"), "img": img,
            })
    pool.append(entry)

json.dump(pool, open(OUT / "web/pool.json", "w"), ensure_ascii=False, indent=1)
print(sum(len(p["items"]) for p in pool), "items over", len(pool), "queries")
