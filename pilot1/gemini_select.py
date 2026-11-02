#!/usr/bin/env python3
"""Pilot 1: a Gemini model selects EXACTLY K of the (up to) 10 candidates for the 20 pilot queries x 3 engines.

Prompt in prompt.py; candidates and image preprocessing in ../vlm_subset_selection.py. Calls Google AI Studio's native
generateContent endpoint (Gemma on AI Studio rejects image input) with the key read from ~/.config/gemini_api_key.
Writes pilot1/<model>_k<K>_selection.jsonl and resumes from it.

Usage: python gemini_select.py --k 3 --model gemini-3.1-flash-lite
"""
import argparse, json, os, sys, time, urllib.error, urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import vlm_subset_selection as base  # noqa: E402
from prompt import PROMPT_VERSION, QUERY_IDS, make_prompt  # noqa: E402


def api_key():
    return open(os.path.expanduser("~/.config/gemini_api_key")).read().strip()


def call(model, k, query, engine, its):
    prompt = make_prompt(k, query, engine, len(its))
    parts = [{"text": prompt}]
    for p, it in enumerate(its, 1):
        crop = it.get("crop_path"); path = base.ROOT / "parsed" / crop if crop else None
        if path and path.exists():
            parts.append({"text": f"POSITION {p}; title={it.get('title')}; price={it.get('price')}; sponsored={it.get('sponsored', False)}"})
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": base.img_url(path).split(",", 1)[1]}})
    payload = {"contents": [{"role": "user", "parts": parts}], "generationConfig": {"maxOutputTokens": 8192, "responseMimeType": "application/json"}}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"x-goog-api-key": api_key(), "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read().decode())
    text = "".join(p.get("text", "") for p in result["candidates"][0]["content"]["parts"] if not p.get("thought"))
    parsed = base.parse_json(text)
    sel = parsed.get("selected_positions", [])
    if not isinstance(sel, list) or len(sel) != min(k, len(its)) or len(set(sel)) != len(sel) or any(not isinstance(x, int) or x < 1 or x > len(its) for x in sel):
        raise ValueError(f"invalid selection {sel}")
    return sel, parsed.get("relevance_by_position"), parsed.get("selection_notes", ""), result.get("usageMetadata"), result.get("modelVersion")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, required=True); ap.add_argument("--model", default="gemini-3.1-flash-lite")
    a = ap.parse_args()
    out = HERE / f"{a.model}_k{a.k}_selection.jsonl"
    rs = base.rows()
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            r = json.loads(l)
            if r.get("status") == "ok": done.add((r["query_id"], r["engine"]))
    tasks = [(q, e) for q in QUERY_IDS for e in base.ENGINES if (q, e) not in done]
    with open(out, "a") as fh:
        for n, (q, e) in enumerate(tasks, 1):
            its = base.items(rs[str(q)], e)
            for attempt in range(1, 5):
                try:
                    sel, rel, notes, usage, version = call(a.model, a.k, rs[str(q)]["query"], e, its)
                    fh.write(json.dumps({"query_id": q, "engine": e, "k": a.k, "prompt_version": PROMPT_VERSION, "model": a.model, "model_version": version, "n_candidates": len(its),
                                         "selected_positions": sel, "relevance_by_position": rel, "selection_notes": notes, "status": "ok",
                                         "attempt": attempt, "usage": usage, "finished_at": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False) + "\n")
                    fh.flush(); print(f"[{n}/{len(tasks)}] {q}/{e} selected={sel}", flush=True); break
                except urllib.error.HTTPError as ex:
                    body = ex.read().decode()[:200].replace("\n", " ")
                    print(f"[{n}/{len(tasks)}] {q}/{e} attempt {attempt} HTTP {ex.code}: {body}", flush=True)
                    time.sleep(65 if ex.code == 429 else 10 * attempt)
                except Exception as ex:
                    print(f"[{n}/{len(tasks)}] {q}/{e} attempt {attempt} failed: {type(ex).__name__}: {str(ex)[:200]}", flush=True)
                    time.sleep(10 * attempt)
            else:
                fh.write(json.dumps({"query_id": q, "engine": e, "k": a.k, "model": a.model, "selected_positions": [], "status": "failed_after_retries"}) + "\n"); fh.flush()
            time.sleep(4)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
