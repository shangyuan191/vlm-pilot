"""Pilot 2 summary over candidate orders: mean (and population std) of the per-order scores of score_pilot2.py.

Covers every output that exists: the original full/caption/blind, the 2026-11-03 extra conditions
(stylename / fulldesc / fullmark), fixed-caption runs (<model>_cap-<source>), the English prompt (<model>_en),
and the order-independent per-image pipeline (pipe-pred / pipe-gold / pipe-soft, one run only).

Usage: python summarize_pilot2.py
"""
import json, statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = ["qwen3-vl-8b-bnb4", "qwen3-vl-32b-bnb4", "gemini-3.1-flash-lite", "internvl3-8b-bnb4", "llava-ov-7b-fp8"]
CONDS = ["full", "caption", "blind", "stylename", "fulldesc", "fullmark"]
labels = json.load(open(HERE / "human_labels.json"))
qs = {q["qid"] for q in json.load(open(HERE / "questions.json")) if labels[q["qid"]]["query"] != "N"}


def score(path):
    if not path.exists(): return None
    rows = [r for r in map(json.loads, path.read_text().splitlines()) if r.get("status") == "ok" and r["qid"] in qs]
    return sum(sum(labels[r["qid"]]["items"][str(p)] == "P" for p in r["selected_positions"]) for r in rows) / len(rows), len(rows)


def orders(tag, cond):
    s = [score(HERE / f"{tag}{o}_{cond}.jsonl") for o in ("", "_o1", "_o2")]
    if None in s: return "—"
    v = [x[0] for x in s]
    return f"{st.mean(v):.2f} ±{st.pstdev(v):.2f}" + ("" if all(x[1] == 20 for x in s) else f" (n={'/'.join(str(x[1]) for x in s)})")


print("## Conditions (3-order mean ± std)\n")
print("| Model | " + " | ".join(CONDS) + " |"); print("|---" * (len(CONDS) + 1) + "|")
for m in MODELS: print(f"| {m} | " + " | ".join(orders(m, c) for c in CONDS) + " |")

print("\n## Caption condition with fixed captions (3-order mean ± std)\n")
srcs = ["claude-opus-5-5", "qwen3-vl-32b-bnb4"]
print("| Selector | own captions | " + " | ".join(srcs) + " |"); print("|---|---|---|---|")
for m in MODELS: print(f"| {m} | {orders(m, 'caption')} | " + " | ".join(orders(f'{m}_cap-{s}', 'caption') for s in srcs) + " |")

print("\n## English prompt (qwen3-vl-8b-bnb4_en)\n")
print(" | ".join(f"{c} {orders('qwen3-vl-8b-bnb4_en', c)}" for c in ("full", "caption", "blind")))

print("\n## Per-image pipeline (order-independent, one run)\n")
print("| Model | pipe-pred | pipe-gold | pipe-soft |"); print("|---|---|---|---|")
for m in MODELS:
    s = [score(HERE / f"{m}_pipe-{c}.jsonl") for c in ("pred", "gold", "soft")]
    if None not in s: print(f"| {m} | " + " | ".join(f"{x[0]:.2f}" for x in s) + " |")
