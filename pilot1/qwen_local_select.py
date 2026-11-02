#!/usr/bin/env python3
"""Pilot 1: a local vLLM model (OpenAI-compatible server) selects EXACTLY K candidates.

Prompt in prompt.py; candidates and image preprocessing in ../vlm_subset_selection.py.
Writes pilot1/<served-model>_k<K>_selection.jsonl and resumes from it.

Usage: python qwen_local_select.py --k 3 [--model qwen3-vl-8b-bnb4] [--url http://127.0.0.1:8021/v1/chat/completions]
"""
import argparse, json, sys, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import vlm_subset_selection as base  # noqa: E402
from prompt import PROMPT_VERSION, QUERY_IDS, make_prompt  # noqa: E402


def call(url, model, k, query, engine, its):
    content = [{"type": "text", "text": make_prompt(k, query, engine, len(its))}]
    for p, it in enumerate(its, 1):
        crop = it.get("crop_path"); path = base.ROOT / "parsed" / crop if crop else None
        if path and path.exists():
            content.append({"type": "text", "text": f"POSITION {p}; title={it.get('title')}; price={it.get('price')}; sponsored={it.get('sponsored', False)}"})
            content.append({"type": "image_url", "image_url": {"url": base.img_url(path)}})
    payload = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": 600, "temperature": 0}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=600) as resp:
        result = json.loads(resp.read().decode())
    parsed = base.parse_json(result["choices"][0]["message"].get("content") or "")
    sel = parsed.get("selected_positions", [])
    if not isinstance(sel, list) or len(sel) != min(k, len(its)) or len(set(sel)) != len(sel) or any(not isinstance(x, int) or x < 1 or x > len(its) for x in sel):
        raise ValueError(f"invalid selection {sel}")
    return sel, parsed.get("relevance_by_position"), parsed.get("selection_notes", ""), result.get("usage")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True); ap.add_argument("--model", default="qwen3-vl-8b-bnb4")
    ap.add_argument("--url", default="http://127.0.0.1:8021/v1/chat/completions")
    a = ap.parse_args()
    out = HERE / f"{a.model}_k{a.k}_selection.jsonl"
    rs = base.rows(); done = set()
    if out.exists():
        done = {(r["query_id"], r["engine"]) for r in map(json.loads, out.read_text().splitlines()) if r.get("status") == "ok"}
    tasks = [(q, e) for q in QUERY_IDS for e in base.ENGINES if (q, e) not in done]
    with open(out, "a") as fh:
        for n, (q, e) in enumerate(tasks, 1):
            its = base.items(rs[str(q)], e)
            for attempt in range(1, 4):
                try:
                    sel, rel, notes, usage = call(a.url, a.model, a.k, rs[str(q)]["query"], e, its)
                    fh.write(json.dumps({"query_id": q, "engine": e, "k": a.k, "prompt_version": PROMPT_VERSION, "model": a.model, "n_candidates": len(its), "selected_positions": sel,
                                         "relevance_by_position": rel, "selection_notes": notes, "status": "ok", "attempt": attempt, "usage": usage,
                                         "finished_at": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False) + "\n")
                    fh.flush(); print(f"[{n}/{len(tasks)}] {q}/{e} selected={sel}", flush=True); break
                except Exception as ex:
                    print(f"[{n}/{len(tasks)}] {q}/{e} attempt {attempt} failed: {type(ex).__name__}: {str(ex)[:200]}", flush=True)
                    time.sleep(3)
            else:
                fh.write(json.dumps({"query_id": q, "engine": e, "k": a.k, "model": a.model, "selected_positions": [], "status": "failed_after_retries"}) + "\n"); fh.flush()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
