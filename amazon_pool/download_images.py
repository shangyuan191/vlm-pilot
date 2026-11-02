"""Download the product images referenced by pilot2/questions.json and amazon_pool/pools.json
into data/derived/amazon_images/<asin>.jpg (skips files already present)."""
import json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data/derived/amazon_images"; DEST.mkdir(parents=True, exist_ok=True)
want = {}
for q in json.load(open(ROOT / "pilot2/questions.json")):
    want[q["query_item"]["asin"]] = q["query_item"]["img"]
    for c in q["candidates"]: want[c["asin"]] = c["img"]
for p in json.load(open(ROOT / "amazon_pool/pools.json")):
    for i in p["items"]: want[i["asin"]] = i["img"]


def get(item):
    asin, url = item; f = DEST / f"{asin}.jpg"
    if f.exists() and f.stat().st_size > 0: return "skip"
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research dataset preview)"})
            f.write_bytes(urllib.request.urlopen(req, timeout=30).read()); time.sleep(0.2); return "ok"
        except Exception as e:
            err = f"{type(e).__name__}: {e}"; time.sleep(2 * (attempt + 1))
    return "fail " + asin + " " + err[:80]


with ThreadPoolExecutor(4) as ex:
    res = list(ex.map(get, want.items()))
print(len(want), "images:", {k: sum(r.startswith(k) for r in res) for k in ("ok", "skip", "fail")})
for r in res:
    if r.startswith("fail"): print(r)
