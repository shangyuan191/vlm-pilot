"""Pilot 2 "explicit pipeline" check (added 2026-11-04).

Instead of one call that sees the reference furniture plus 20 lamps, the model judges each image alone:
  step 1: the reference furniture image  -> a 0-10 fit score for each of the 5 styles
  step 2: each lamp image (261 distinct)  -> the same 5 scores
  step 3: no model call. Pick 3 lamps by rule:
     pred : top style of step 1, lamps ranked by their score for that style
     gold : the true style (compare with the "stylename" condition)
     soft : lamps ranked by the dot product of the two score vectors
  Ties are broken by the soft score, then by a fixed hash of the ASIN (never by position).
Each image is judged in its own call, so candidate order cannot matter and no order seeds are needed.

Usage: python run_pipeline.py --model qwen3-vl-32b-bnb4 --url http://127.0.0.1:8041/v1/chat/completions
       python run_pipeline.py --backend gemini --model gemini-3.1-flash-lite
Writes <model>_pipe_judgements.jsonl (resumable) and <model>_pipe-{pred,gold,soft}.jsonl (score_pilot2-compatible rows).
"""
import argparse, hashlib, json, time
from pathlib import Path
import run_pilot2 as rp

HERE = Path(__file__).resolve().parent
STYLES = list(rp.STYLE_NAME)                      # mid-century, farmhouse, industrial, coastal, scandinavian
LIST = "、".join(f"「{rp.STYLE_NAME[s]}」" for s in STYLES)
PROMPT = {
    "query": "這是一件家具的商品圖片。",
    "lamp": "這是一盞燈具的商品圖片。",
}
TAIL = (f"請判斷它屬於以下 5 種室內設計風格的程度：{LIST}。"
        "每種風格給 0–10 分（10 = 非常典型的這種風格，0 = 完全不是），分數可以並列，也可以都很低。"
        '只輸出 JSON，格式：{"scores": {"mid-century modern": 分數, "farmhouse": 分數, "industrial": 分數, '
        '"coastal": 分數, "Scandinavian": 分數}, "reason": "一句話理由"}')


def judge(url, model, asin, kind):
    content = [{"type": "image_url", "image_url": {"url": rp.img_url(asin)}}, {"type": "text", "text": PROMPT[kind] + TAIL}]
    text, usage = rp.post(url, model, content, 300)
    p = rp.parse_json(text)
    sc = {s: float(p["scores"][rp.STYLE_NAME[s]]) for s in STYLES}
    return sc, p.get("reason", ""), usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--url", default="http://127.0.0.1:8041/v1/chat/completions")
    ap.add_argument("--backend", choices=["openai", "gemini"], default="openai"); a = ap.parse_args()
    rp.BACKEND = a.backend
    qs = json.load(open(HERE / "questions.json"))
    todo = [(q["query_item"]["asin"], "query") for q in qs]
    seen = set()
    for q in qs:
        for c in q["candidates"]:
            if c["asin"] not in seen: seen.add(c["asin"]); todo.append((c["asin"], "lamp"))

    jf = HERE / f"{a.model}_pipe_judgements.jsonl"
    have = {(r["asin"], r["kind"]): r for r in map(json.loads, jf.read_text().splitlines())} if jf.exists() else {}
    failed = 0
    with open(jf, "a") as f:
        for i, (asin, kind) in enumerate(todo):
            if (asin, kind) in have: continue
            for att in range(1, 4):
                try:
                    sc, why, usage = judge(a.url, a.model, asin, kind)
                    r = {"asin": asin, "kind": kind, "scores": sc, "reason": why, "usage": usage, "attempt": att}
                    have[(asin, kind)] = r; f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush(); break
                except Exception as e:
                    print(f"{asin} {kind} attempt {att}: {type(e).__name__} {str(e)[:120]}", flush=True)
                    time.sleep(10 if a.backend == "gemini" else 2)
            else: failed += 1
            if a.backend == "gemini": time.sleep(4)
            if i % 50 == 0: print(f"{i}/{len(todo)} judged", flush=True)

    # step 3: rule-based selection
    h = lambda asin: hashlib.md5(asin.encode()).hexdigest()
    outs = {m: [] for m in ("pred", "gold", "soft")}
    for q in qs:
        qj = have.get((q["query_item"]["asin"], "query"))
        cands = [c for c in q["candidates"] if (c["asin"], "lamp") in have]
        if not qj or len(cands) < len(q["candidates"]):
            for m in outs: outs[m].append({"qid": q["qid"], "status": "failed_after_judging"})
            continue
        qv = qj["scores"]; pred = max(STYLES, key=lambda s: (qv[s], -STYLES.index(s)))
        lv = {c["pos"]: have[(c["asin"], "lamp")]["scores"] for c in cands}
        soft = {c["pos"]: sum(qv[s] * lv[c["pos"]][s] for s in STYLES) for c in cands}
        key = {
            "pred": lambda c: (-lv[c["pos"]][pred], -soft[c["pos"]], h(c["asin"])),
            "gold": lambda c: (-lv[c["pos"]][q["style"]], -soft[c["pos"]], h(c["asin"])),
            "soft": lambda c: (-soft[c["pos"]], h(c["asin"])),
        }
        for m in outs:
            sel = [c["pos"] for c in sorted(cands, key=key[m])[:rp.K]]
            outs[m].append({"qid": q["qid"], "condition": f"pipe-{m}", "model": a.model, "k": rp.K, "selected_positions": sel,
                            "predicted_query_style": pred, "query_scores": qv, "status": "ok"})
    for m, rows in outs.items():
        (HERE / f"{a.model}_pipe-{m}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"done: {len(have)} judgements, {failed} failed")


if __name__ == "__main__":
    main()
