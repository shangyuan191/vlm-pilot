"""Pilot 2 (style alignment, Level 1): build questions from the Amazon index (metadata only, no network).

Question format: a query image of a living-room chair (or, when a style has too few,
a coffee/end table) in style S + the text
"lamps in this style"; 20 candidate lamps = 5 same-style + 5 near-miss + 10 random, the model picks 3.
  same-style : lamps whose title names style S (and no other style)
  near-miss  : lamps of another style that share the chair's colour word, so colour alone misleads
  random     : lamps of another style
Style comes from title keywords, so every question still needs a human check of the images.
Candidate titles are given to the model with all style words removed (title_clean), so the style
cannot be read off the text; this is what the Blind / Caption / Full comparison relies on.

Output: pilot2/questions.json
"""
import gzip, json, random, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
STYLES = {"mid-century": r"mid[\s-]?century|\bmcm\b", "farmhouse": r"farmhouse", "industrial": r"industrial",
          "coastal": r"coastal|nautical", "scandinavian": r"scandinavian|nordic"}
OTHER_STYLE_WORDS = r"boho|bohemian|art deco|\bglam\b|japanese|japandi|wabi[\s-]?sabi|rustic|vintage|retro|modern|contemporary|traditional|transitional"
COLORS = ["black", "white", "gray", "grey", "brown", "beige", "cream", "gold", "brass", "silver", "green", "blue", "navy", "teal", "pink", "red",
          "yellow", "orange", "walnut", "oak", "wood", "natural"]
LAMPS = {"Table Lamps", "Floor Lamps", "Pendant Lights"}
QUERY_LEAVES = ["Chairs", "Coffee Tables", "End Tables"]  # chairs first; tables fill styles with few chairs
ACCESSORY = r"cord kit|bulb guard|\bcage\b|replacement|shade only|socket|\bkit\b|bulbs? only|canopy|adapter|\bpack\b of (bulbs|cages)|lamp ?shades? for"
TOP_N = 40  # sample same-style / near-miss / random lamps from the TOP_N most-reviewed of each group
PER_STYLE = 4
SAME, NEAR, TOTAL = 5, 5, 20


def styles_of(t):
    return [s for s, p in STYLES.items() if re.search(p, t)]


def colors_of(r):
    t = f"{r.get('title') or ''} {r.get('color') or ''}".lower()
    return {("gray" if c == "grey" else c) for c in COLORS if re.search(rf"\b{c}\b", t)}


def clean(title):
    t = re.sub("|".join(STYLES.values()) + "|" + OTHER_STYLE_WORDS, "", title, flags=re.I)
    return re.sub(r"\s{2,}", " ", re.sub(r"\s+([,.;)])", r"\1", t)).strip(" ,-")


def main():
    rng = random.Random(20261101)
    rows = [json.loads(l) for l in gzip.open(ROOT / "data/derived/amazon_index.jsonl.gz", "rt")]
    chairs, lamps = {s: [] for s in STYLES}, {s: [] for s in STYLES}
    for r in rows:
        if not r["cats"]: continue
        t = (r.get("title") or "").lower()
        ss = styles_of(t)
        if len(ss) != 1: continue
        if r["cats"][-1] in QUERY_LEAVES and "Living Room Furniture" in r["cats"]: chairs[ss[0]].append(r)
        elif r["cats"][-1] in LAMPS and not re.search(ACCESSORY, t): lamps[ss[0]].append(r)
    pop = lambda r: int(str(r.get("rating_number") or 0).replace(",", "") or 0)
    top = lambda xs: sorted(xs, key=pop, reverse=True)[:TOP_N]
    questions = []
    for s in STYLES:
        qs = sorted(chairs[s], key=lambda r: (QUERY_LEAVES.index(r["cats"][-1]), -pop(r)))
        stores, picked = set(), []
        for r in qs:  # most-reviewed chairs, at most one per store
            if r.get("store") not in stores: picked.append(r); stores.add(r.get("store"))
            if len(picked) == PER_STYLE: break
        for qn, chair in enumerate(picked, 1):
            ccol = colors_of(chair)
            same = rng.sample(top(lamps[s]), SAME)
            others = [l for t in STYLES if t != s for l in top(lamps[t])]
            near_pool = [l for t in STYLES if t != s for l in top([l for l in lamps[t] if colors_of(l) & ccol])]
            near = rng.sample(near_pool, min(NEAR, len(near_pool)))
            used = {l["asin"] for l in same + near}
            rand = rng.sample([l for l in others if l["asin"] not in used], TOTAL - len(same) - len(near))
            cands = [(l, "same-style") for l in same] + [(l, "near-miss") for l in near] + [(l, "random") for l in rand]
            rng.shuffle(cands)
            questions.append({
                "qid": f"{s}-{qn}", "style": s, "text": "lamps in this style",
                "query_item": {"asin": chair["asin"], "category": chair["cats"][-1], "title": chair["title"], "color": chair.get("color"), "img": chair["img"], "colors": sorted(ccol)},
                "candidates": [{"pos": i, "asin": l["asin"], "title": l["title"], "title_clean": clean(l["title"]), "img": l["img"], "bucket": b,
                                "keyword_style": styles_of(l["title"].lower())[0], "colors": sorted(colors_of(l))} for i, (l, b) in enumerate(cands, 1)]})
    json.dump(questions, open(OUT / "questions.json", "w"), ensure_ascii=False, indent=1)
    print(len(questions), "questions;", "lamps per style:", {s: len(v) for s, v in lamps.items()}, "chairs per style:", {s: len(v) for s, v in chairs.items()})
    for q in questions[::4]:
        print("\n##", q["qid"], "|", q["query_item"]["title"][:90], "| colors", q["query_item"]["colors"])
        for c in q["candidates"][:6]: print("  ", c["bucket"].ljust(10), c["keyword_style"].ljust(12), "|", c["title_clean"][:80])


if __name__ == "__main__":
    main()
