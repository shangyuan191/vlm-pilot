"""Pilot 2: add 7 questions whose query images were picked by eye (2 farmhouse, 2 coastal, 3 scandinavian),
so every style has 4 usable questions. Candidates are drawn exactly as in build_pilot2.py; the existing
20 questions are left untouched (a copy is kept in questions_v1_20q.json). Appends to questions.json.
"""
import gzip, json, random, re
from pathlib import Path
from build_pilot2 import ACCESSORY, LAMPS, NEAR, SAME, STYLES, TOP_N, TOTAL, clean, colors_of, styles_of

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
PICKED = {"farmhouse": ["B091HZYXKX", "B085DVN2TF"], "coastal": ["B01MQ0XY7Q", "B07TC53JSY"],
          "scandinavian": ["B07WQFJT4J", "B0C598517S", "B09L6CXZ65"]}


def main():
    rng = random.Random(20261102)
    rows = [json.loads(l) for l in gzip.open(ROOT / "data/derived/amazon_index.jsonl.gz", "rt")]
    by_asin = {r["asin"]: r for r in rows}
    lamps = {s: [] for s in STYLES}
    for r in rows:
        if not r["cats"] or r["cats"][-1] not in LAMPS: continue
        t = (r.get("title") or "").lower(); ss = styles_of(t)
        if len(ss) == 1 and not re.search(ACCESSORY, t): lamps[ss[0]].append(r)
    pop = lambda r: int(str(r.get("rating_number") or 0).replace(",", "") or 0)
    top = lambda xs: sorted(xs, key=pop, reverse=True)[:TOP_N]
    qs = json.load(open(HERE / "questions.json"))
    for s, asins in PICKED.items():
        start = sum(q["style"] == s for q in qs)
        for n, asin in enumerate(asins, start + 1):
            chair = by_asin[asin]; ccol = colors_of(chair)
            same = rng.sample(top(lamps[s]), SAME)
            others = [l for t in STYLES if t != s for l in top(lamps[t])]
            near_pool = [l for t in STYLES if t != s for l in top([l for l in lamps[t] if colors_of(l) & ccol])]
            near = rng.sample(near_pool, min(NEAR, len(near_pool)))
            used = {l["asin"] for l in same + near}
            rand = rng.sample([l for l in others if l["asin"] not in used], TOTAL - len(same) - len(near))
            cands = [(l, "same-style") for l in same] + [(l, "near-miss") for l in near] + [(l, "random") for l in rand]
            rng.shuffle(cands)
            qs.append({"qid": f"{s}-{n}", "style": s, "text": "lamps in this style", "added": "2026-11-01 (query picked by eye)",
                       "query_item": {"asin": asin, "category": chair["cats"][-1], "title": chair["title"], "color": chair.get("color"), "img": chair["img"], "colors": sorted(ccol)},
                       "candidates": [{"pos": i, "asin": l["asin"], "title": l["title"], "title_clean": clean(l["title"]), "img": l["img"], "bucket": b,
                                       "keyword_style": styles_of(l["title"].lower())[0], "colors": sorted(colors_of(l))} for i, (l, b) in enumerate(cands, 1)]})
            print("added", qs[-1]["qid"], asin, "colors", sorted(ccol))
    json.dump(qs, open(HERE / "questions.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
