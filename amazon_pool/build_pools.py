"""Prototype: larger candidate pools from Amazon for Pilot 1 queries (metadata only, no network).

For each query we map it to one Amazon leaf category and write keyword rules for its conditions
(applied to title + color/material/style fields). Every item in the category is tagged with which
conditions it meets, then a 50-item pool is sampled:
  up to 15 "match"     items meeting every rule,
  up to 15 "near-miss" items failing exactly one rule (spread over the rules),
  the rest  "random"   items from the same category.
Rule matches are only a proxy: the pool still needs human labels, like Pilot 1.

Output: amazon_pool/pools.json and amazon_pool/pool_stats.md
"""
import gzip, json, random, re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
POOL = {"match": 15, "near": 15, "total": 50}
BEIGE = r"beige|cream|ivory|\btan\b|taupe|khaki|oatmeal|natural|sand|linen"

# (query_id, Subtext query, category leaf, [(condition name, regex, must_match)])
QUERIES = [
    (11, "moody dark-green velvet armchair for a small apartment, no roll arms, no tufting", "Chairs|Living Room Furniture",
     [("green", r"dark green|emerald|forest green|hunter green|olive|deep green|\bgreen\b", True), ("velvet", r"velvet", True),
      ("armchair", r"arm ?chair|accent chair|club chair|lounge chair", True),
      ("no roll arms", r"roll(ed)? arm", False), ("no tufting", r"tuft|button|chesterfield", False)]),
    (28, "rug that hides cat puke but isn't beige", "Area Rugs|",
     [("patterned", r"pattern|abstract|distressed|vintage|persian|oriental|geometric|floral|medallion|traditional|moroccan|boho", True),
      ("not beige", BEIGE, False)]),
    (38, "armchair that swivels, has a low back, isn't a club chair, isn't a barrel chair, in leather", "Chairs|Living Room Furniture",
     [("swivel", r"swivel", True), ("leather", r"leather", True), ("not club", r"\bclub\b", False), ("not barrel", r"barrel", False)]),
    (44, "couch that looks like a pair of pink lips", "Sofas & Couches|",
     [("lips", r"\blips?\b|mouth|kiss", True), ("pink", r"pink|rose|blush|fuchsia|magenta", True)]),
    (72, "memphis Group side table, terrazzo top, squiggle legs", "End Tables|",
     [("terrazzo", r"terrazzo", True), ("memphis", r"memphis|postmodern|80s|1980", True), ("squiggle legs", r"squiggl|wavy|curv|zig ?zag", True)]),
    (74, "1970s Italian travertine coffee table, oval, chunky", "Coffee Tables|",
     [("travertine", r"travertine", True), ("oval", r"\boval\b", True), ("chunky", r"chunky|thick|solid|plinth|pedestal|block", True)]),
    (75, "postmodern floor lamp in candy pink with a globe shade", "Floor Lamps|",
     [("pink", r"pink|blush|rose", True), ("globe", r"globe|sphere|orb|\bball\b", True)]),
    (76, "wabi-sabi ceramic vase, wood-fired, asymmetric, oxblood glaze", "Vases|",
     [("ceramic", r"ceramic|pottery|stoneware|porcelain|clay", True), ("oxblood/red", r"oxblood|ox blood|burgundy|\bred\b|crimson|maroon", True),
      ("wabi-sabi/irregular", r"wabi|irregular|asymmetric|rustic|handmade|primitive|raw", True)]),
    (86, "ergonomic task chair in a real color, not all-black mesh — walnut frame, cognac leather", "Home Office Desk Chairs|",
     [("wood frame", r"walnut|wood", True), ("cognac/brown leather", r"cognac|saddle|\btan\b|caramel|brown", True), ("leather", r"leather", True),
      ("not mesh", r"mesh", False)]),
]


def text(r):
    return " ".join(str(r.get(k) or "") for k in ("title", "color", "material", "style", "shape")).lower()


def main():
    rows = [json.loads(l) for l in gzip.open(ROOT / "data/derived/amazon_index.jsonl.gz", "rt")]
    rng = random.Random(20261101)
    pools, lines = [], ["| Query | Amazon 類別 | 類別商品數 | 符合所有規則 | 只差一條（near-miss） |", "|---|---|---:|---:|---:|"]
    for qid, query, cat, rules in QUERIES:
        leaf, parent = cat.split("|")
        items = [r for r in rows if r["cats"] and r["cats"][-1] == leaf and (not parent or parent in r["cats"])]
        tagged = []
        for r in items:
            t = text(r)
            ok = [bool(re.search(p, t)) == must for _, p, must in rules]
            tagged.append((r, ok))
        match = [r for r, ok in tagged if all(ok)]
        near = [(r, ok.index(False)) for r, ok in tagged if ok.count(False) == 1]
        rest = [r for r, ok in tagged if ok.count(False) >= 2]
        lines.append(f"| {qid} {query[:45]}… | {leaf} | {len(items)} | {len(match)} | {len(near)} |")
        rng.shuffle(match); rng.shuffle(rest)
        by_rule = {}
        for r, i in near: by_rule.setdefault(i, []).append(r)
        for v in by_rule.values(): rng.shuffle(v)
        near_pick = []
        while len(near_pick) < POOL["near"] and any(by_rule.values()):
            for i in sorted(by_rule):
                if by_rule[i] and len(near_pick) < POOL["near"]:
                    near_pick.append((by_rule[i].pop(), rules[i][0]))
        pick = [(r, "match", None) for r in match[:POOL["match"]]] + [(r, "near-miss", why) for r, why in near_pick]
        pick += [(r, "random", None) for r in rest[:POOL["total"] - len(pick)]]
        rng.shuffle(pick)
        pools.append({"query_id": qid, "query": query, "category": leaf, "rules": [{"name": n, "regex": p, "must_match": m} for n, p, m in rules],
                      "category_size": len(items), "n_match": len(match), "n_near_miss": len(near),
                      "items": [{"asin": r["asin"], "title": r["title"], "price": r.get("price"), "color": r.get("color"), "material": r.get("material"),
                                 "style": r.get("style"), "img": r["img"], "bucket": b, "fails_rule": why} for r, b, why in pick]})
    json.dump(pools, open(OUT / "pools.json", "w"), ensure_ascii=False, indent=1)
    (OUT / "pool_stats.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
