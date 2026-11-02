"""Pilot 1 at a different K, using only methods that need no model call:
Oracle-K, Top-K, and the exact expectation of Random-K (average over every K-subset of the pool).

Usage (inside pilot1/): python score_k.py <labels.json> <K> [name=selection.jsonl ...]
Each selection file has one {"query_id", "engine", "selected_positions", "status"} row per pool.
"""
import json, sys, itertools

ENG = {"amazon": "a", "google_shopping": "g", "onton": "o"}


def diversity(items, dims):
    """Mean over dimensions of (#distinct - 1) / (min(n, #options) - 1); None if fewer than 2 items."""
    if len(items) < 2: return None
    vals = []
    for name, opts in dims.items():
        cap = min(len(items), len(opts))
        if cap < 2: continue
        vals.append((len({it["dims"].get(name) for it in items}) - 1) / (cap - 1))
    return sum(vals) / len(vals) if vals else None


def load_sel(specs):
    out = {}
    for spec in specs:
        name, f = spec.split("=", 1)
        out[name] = {(r["query_id"], r["engine"]): r["selected_positions"] for r in map(json.loads, open(f)) if r.get("status") == "ok"}
    return out


def main(path, K, specs=()):
    labels = json.load(open(path))
    sels = load_sel(specs)
    tax = {r["query_id"]: r["tags"] for r in json.load(open(__import__("pathlib").Path(__file__).resolve().parent.parent / "query_taxonomy.json"))["rows"]}
    rows = []
    for q, lab in labels.items():
        q = int(q)
        for eng, e in ENG.items():
            pool = {int(c[1:]): lab["items"][c] for c in lab["items"] if c[0] == e}
            if len(pool) < K: continue
            items = list(pool.values())
            stats = []
            for combo in itertools.combinations(items, K):
                p = [it for it in combo if it["label"] == "P"]
                stats.append((len(p), diversity(p, lab["dims"])))
            best = max(stats, key=lambda s: (s[0], -1 if s[1] is None else s[1]))
            rdiv = [d for _, d in stats if d is not None]
            top = [pool[i] for i in sorted(pool)[:K]]
            tp = [it for it in top if it["label"] == "P"]
            extra = {}
            for name, sel in sels.items():
                if (q, eng) not in sel: continue
                sp = [pool[i] for i in sel[(q, eng)] if i in pool and pool[i]["label"] == "P"]
                extra[name] = len(sp); extra[name + "_div"] = diversity(sp, lab["dims"])
            rows.append({**extra, "q": q, "eng": eng, "pool_pass": sum(it["label"] == "P" for it in items),
                         "Oracle": best[0], "Oracle_div": best[1], "Top": len(tp), "Top_div": diversity(tp, lab["dims"]),
                         "Random": sum(s[0] for s in stats) / len(stats), "Random_div": sum(rdiv) / len(rdiv) if rdiv else None})
    n = len(rows)
    avg = lambda k: sum(r[k] for r in rows) / n
    davg = lambda k: (lambda v: sum(v) / len(v) if v else float("nan"))([r[k] for r in rows if r[k] is not None])
    print(f"K={K}, pools={n}")
    for name in sels:
        have = [r for r in rows if name in r]
        if len(have) != n: print(f"  NOTE: {name} covers {len(have)}/{n} pools; the averages below use only these pools")
        rows = have; n = len(rows)
    for m in ("Oracle", *sels, "Top", "Random"):
        print(f"  {m:7s} pass {avg(m):.2f}/{K}  ({avg(m)/K:.0%})   diversity {davg(m + '_div'):.3f}")
    print(f"  pools with fewer than {K} passing items: {sum(r['pool_pass'] < K for r in rows)}/{n}; with 0: {sum(r['pool_pass'] == 0 for r in rows)}")
    print(f"  Oracle-Random gap = {avg('Oracle') - avg('Random'):.2f}")
    for name in sels:
        print(f"  {name} closes {(avg(name) - avg('Random')) / (avg('Oracle') - avg('Random')):.0%} of the Random->Oracle gap; "
              f"equals Oracle in {sum(r[name] == r['Oracle'] for r in rows)}/{n} pools")
    for tag in ("negation", "multi_constraint", "subjective_aesthetic", "cultural_or_style_reference"):
        sel = [r for r in rows if tag in tax[r["q"]]]
        extra = "".join(f"  {name} {sum(r[name] for r in sel)/len(sel):.2f}" for name in sels)
        print(f"  {tag:28s} Oracle {sum(r['Oracle'] for r in sel)/len(sel):.2f}{extra}  Random {sum(r['Random'] for r in sel)/len(sel):.2f}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3:])
