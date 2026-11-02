"""Score Pilot 2 selections against the human labels.

For each condition: mean number of same-style lamps among the 3 picks, the Random expectation
(3 x share of same-style lamps), how often a pick is a colour-matched near-miss that the annotator
marked "not same style", and the same numbers per style.

Usage: python score_pilot2.py [--model qwen3-vl-8b-bnb4]
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="qwen3-vl-8b-bnb4"); a = ap.parse_args()
    labels = json.load(open(HERE / "human_labels.json"))
    qs = {q["qid"]: q for q in json.load(open(HERE / "questions.json")) if q["qid"] in labels and labels[q["qid"]]["query"] != "N"}
    lab = lambda qid, pos: labels[qid]["items"][str(pos)]
    bucket = lambda qid, pos: next(c["bucket"] for c in qs[qid]["candidates"] if c["pos"] == pos)
    rnd = {qid: 3 * sum(lab(qid, c["pos"]) == "P" for c in q["candidates"]) / len(q["candidates"]) for qid, q in qs.items()}
    oracle = {qid: min(3, sum(lab(qid, c["pos"]) == "P" for c in q["candidates"])) for qid, q in qs.items()}

    rows = {}
    for cond in ("full", "caption", "blind"):
        f = HERE / f"{a.model}_{cond}.jsonl"
        if not f.exists(): continue
        rows[cond] = {r["qid"]: r["selected_positions"] for r in map(json.loads, f.read_text().splitlines()) if r.get("status") == "ok" and r["qid"] in qs}

    print(f"{len(qs)} questions, model {a.model}\n")
    print("| Condition | 3 個裡同風格（平均） | 比例 | 選到「同色但不同風格」的比例 | 完成題數 |")
    print("|---|---:|---:|---:|---:|")
    print(f"| Oracle | {sum(oracle.values())/len(oracle):.2f} | {sum(oracle.values())/len(oracle)/3:.0%} | — | {len(qs)} |")
    for cond, sel in rows.items():
        hits = [sum(lab(q, p) == "P" for p in s) for q, s in sel.items()]
        trap = [sum(bucket(q, p) == "near-miss" and lab(q, p) != "P" for p in s) for q, s in sel.items()]
        print(f"| {cond} | {sum(hits)/len(hits):.2f} | {sum(hits)/len(hits)/3:.0%} | {sum(trap)/(3*len(trap)):.0%} | {len(sel)} |")
    print(f"| Random（期望值） | {sum(rnd.values())/len(rnd):.2f} | {sum(rnd.values())/len(rnd)/3:.0%} | — | {len(qs)} |")

    print("\n| Style | 題數 | " + " | ".join(rows) + " | Random |")
    print("|---|---:|" + "---:|" * (len(rows) + 1))
    by = defaultdict(list)
    for qid, q in qs.items(): by[q["style"]].append(qid)
    for st, ids in by.items():
        cells = []
        for cond, sel in rows.items():
            v = [sum(lab(q, p) == "P" for p in sel[q]) for q in ids if q in sel]
            cells.append(f"{sum(v)/len(v):.2f}" if v else "—")
        print(f"| {st} | {len(ids)} | " + " | ".join(cells) + f" | {sum(rnd[q] for q in ids)/len(ids):.2f} |")

    print("\n每題（同風格數 / 3）：")
    for qid in qs:
        print(" ", qid.ljust(15), " ".join(f"{c}={sum(lab(qid, p) == 'P' for p in rows[c][qid]) if qid in rows[c] else '-'}" for c in rows), f"random={rnd[qid]:.2f}")


if __name__ == "__main__":
    main()
