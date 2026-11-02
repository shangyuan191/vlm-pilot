#!/usr/bin/env python3
"""Pilot 2: Blind / Caption / Full style alignment on the human-validated questions.

All three conditions see the 20 candidate lamps (image + title with style words removed) and the text
"lamps in this style"; they differ only in how the query furniture image is given:
  full    : the query image itself
  caption : a description the same model wrote from the query image (captions are saved first)
  blind   : nothing about the query
The model picks exactly 3 lamps. Questions whose query image the annotator marked "N" are skipped.

Usage: python run_pilot2.py [--model qwen3-vl-8b-bnb4] [--url http://127.0.0.1:8021/v1/chat/completions]
       python run_pilot2.py --backend gemini --model gemini-3.1-flash-lite
(the gemini backend calls Google AI Studio with the key in ~/.config/gemini_api_key)
Writes pilot2/<model>_captions.jsonl and pilot2/<model>_<condition>.jsonl (resumable).

--order-seed S shows the 20 candidates in a different order (random.Random(f"{S}-{qid}")), reuses the
model's captions, and writes pilot2/<model>_o<S>_<condition>.jsonl. selected_positions there are mapped
back to the original positions (so score_pilot2.py works unchanged); shown_positions are what the model saw.
"""
import argparse, base64, io, json, os, random, re, time, urllib.error, urllib.request
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image

HERE = Path(__file__).resolve().parent
IMG = HERE.parent / "data/derived/amazon_images"
K = 3
CAPTION_PROMPT = ("這是一件家具的商品圖片。請用 2–3 句話描述它的設計風格、造型特徵、材質和顏色，"
                  "讓沒看到圖片的人也能想像它的整體風格。不要提到品牌或商品名稱。只輸出描述本身。")


def img_url(asin):
    with Image.open(IMG / f"{asin}.jpg") as im:
        im = im.convert("RGB"); im.thumbnail((900, 900)); b = io.BytesIO(); im.save(b, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


BACKEND = "openai"


def post(url, model, content, max_tokens):
    if BACKEND == "gemini": return post_gemini(model, content)
    payload = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=600) as resp:
        r = json.loads(resp.read().decode())
    return r["choices"][0]["message"].get("content") or "", r.get("usage")


def post_gemini(model, content):
    """Same request as post(), sent to Google AI Studio's generateContent endpoint."""
    parts = [{"text": c["text"]} if c["type"] == "text" else
             {"inline_data": {"mime_type": "image/jpeg", "data": c["image_url"]["url"].split(",", 1)[1]}} for c in content]
    key = open(os.path.expanduser("~/.config/gemini_api_key")).read().strip()
    req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                                 data=json.dumps({"contents": [{"role": "user", "parts": parts}], "generationConfig": {"maxOutputTokens": 8192}}).encode(),
                                 headers={"x-goog-api-key": key, "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=600) as resp:
        r = json.loads(resp.read().decode())
    text = "".join(p.get("text", "") for p in r["candidates"][0]["content"]["parts"] if not p.get("thought"))
    return text, r.get("usageMetadata")


def parse_json(t):
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t.strip(), flags=re.I)
    try: return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, flags=re.S)
        if not m: raise
        return json.loads(m.group(0))


def select(url, model, q, cond, caption):
    if cond == "full":
        head = [{"type": "text", "text": "使用者附上一張參考家具圖片，並說：「lamps in this style」。參考圖片如下："},
                {"type": "image_url", "image_url": {"url": img_url(q["query_item"]["asin"])}}]
    elif cond == "caption":
        head = [{"type": "text", "text": f"使用者附上一張參考家具圖片，並說：「lamps in this style」。參考圖片的文字描述如下：\n{caption}"}]
    else:
        head = [{"type": "text", "text": "使用者說：「lamps in this style」。（使用者提到的參考圖片無法提供。）"}]
    content = head + [{"type": "text", "text": f"以下是 {len(q['candidates'])} 盞候選燈具。請剛好選出 {K} 盞最符合使用者想要的風格的燈。"}]
    for c in q["candidates"]:
        content.append({"type": "text", "text": f"POSITION {c['pos']}; title={c['title_clean']}"})
        content.append({"type": "image_url", "image_url": {"url": img_url(c["asin"])}})
    content.append({"type": "text", "text": "請只回傳 JSON，不要 Markdown。格式如下（尖括號是要你填入的內容，不是答案）："
                    '{"selected_positions":[<position>, ...],"selection_notes":"<一句話說明為什麼這樣選>"}。'
                    f"selected_positions 使用 1-based position，剛好 {K} 個且不得重複；請根據圖片和標題實際判斷後再選。"})
    text, usage = post(url, model, content, 400)
    try: parsed = parse_json(text)
    except json.JSONDecodeError:
        # malformed JSON elsewhere (e.g. an unescaped quote in selection_notes): keep only the selection
        m = re.search(r'"selected_positions"\s*:\s*\[([\d,\s]*)\]', text)
        if not m: raise
        parsed = {"selected_positions": [int(x) for x in re.findall(r"\d+", m.group(1))], "selection_notes": "(notes unparsable) " + text[:300]}
    sel = parsed.get("selected_positions", [])
    if not isinstance(sel, list) or len(sel) != K or len(set(sel)) != K or any(not isinstance(x, int) or not 1 <= x <= len(q["candidates"]) for x in sel):
        raise ValueError(f"invalid selection {sel}")
    return sel, parsed.get("selection_notes", ""), usage


def done_ids(path):
    return {r["qid"] for r in map(json.loads, path.read_text().splitlines()) if r.get("status") == "ok"} if path.exists() else set()


def main():
    global BACKEND
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="qwen3-vl-8b-bnb4"); ap.add_argument("--url", default="http://127.0.0.1:8021/v1/chat/completions")
    ap.add_argument("--backend", choices=["openai", "gemini"], default="openai"); ap.add_argument("--conditions", default="full,caption,blind")
    ap.add_argument("--order-seed", type=int, default=None)
    a = ap.parse_args(); BACKEND = a.backend; conds = a.conditions.split(",")
    labels = json.load(open(HERE / "human_labels.json"))
    qs = [q for q in json.load(open(HERE / "questions.json")) if labels[q["qid"]]["query"] != "N"]
    print(len(qs), "usable questions", flush=True)

    cap_path = HERE / f"{a.model}_captions.jsonl"
    have = done_ids(cap_path) if "caption" in conds else {q["qid"] for q in qs}
    with open(cap_path, "a") as fh:
        for q in qs:
            if q["qid"] in have: continue
            content = [{"type": "image_url", "image_url": {"url": img_url(q["query_item"]["asin"])}}, {"type": "text", "text": CAPTION_PROMPT}]
            for attempt in range(1, 5):
                try:
                    text, usage = post(a.url, a.model, content, 200); break
                except urllib.error.HTTPError as ex:
                    body = ex.read().decode()
                    print("caption", q["qid"], "attempt", attempt, "HTTP", ex.code, body[:120].replace("\n", " "), flush=True)
                    if ex.code == 429 and "PerDay" in body: print("daily quota exhausted, stopping", flush=True); return
                    time.sleep(65 if ex.code == 429 else 20 * attempt)
            else:
                print("caption", q["qid"], "gave up; rerun later", flush=True); continue
            if BACKEND == "gemini": time.sleep(4)
            fh.write(json.dumps({"qid": q["qid"], "caption": text.strip(), "status": "ok", "usage": usage}, ensure_ascii=False) + "\n"); fh.flush()
            print("caption", q["qid"], text.strip()[:100], flush=True)
    captions = {r["qid"]: r["caption"] for r in map(json.loads, cap_path.read_text().splitlines()) if r.get("status") == "ok"} if cap_path.exists() else {}

    tag = a.model if a.order_seed is None else f"{a.model}_o{a.order_seed}"
    for cond in [c for c in ("full", "caption", "blind") if c in conds]:
        out = HERE / f"{tag}_{cond}.jsonl"; have = done_ids(out)
        with open(out, "a") as fh:
            for q in qs:
                if q["qid"] in have: continue
                if cond == "caption" and q["qid"] not in captions: print("caption", q["qid"], "skipped: no caption yet", flush=True); continue
                shown, orig = q, None
                if a.order_seed is not None:
                    cands = list(q["candidates"]); random.Random(f"{a.order_seed}-{q['qid']}").shuffle(cands)
                    orig = [c["pos"] for c in cands]
                    shown = dict(q, candidates=[dict(c, pos=i) for i, c in enumerate(cands, 1)])
                for attempt in range(1, 4):
                    try:
                        sel, notes, usage = select(a.url, a.model, shown, cond, captions.get(q["qid"]))
                        extra = {} if orig is None else {"order_seed": a.order_seed, "shown_positions": sel}
                        if orig is not None: sel = [orig[x - 1] for x in sel]
                        fh.write(json.dumps({"qid": q["qid"], "condition": cond, "model": a.model, "k": K, "selected_positions": sel, **extra, "selection_notes": notes,
                                             "status": "ok", "attempt": attempt, "usage": usage, "finished_at": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False) + "\n")
                        fh.flush(); print(cond, q["qid"], sel, flush=True)
                        if BACKEND == "gemini": time.sleep(6)
                        break
                    except urllib.error.HTTPError as ex:
                        body = ex.read().decode()
                        print(cond, q["qid"], "attempt", attempt, "HTTP", ex.code, body[:150].replace("\n", " "), flush=True)
                        if ex.code == 429 and "PerDay" in body: print("daily quota exhausted, stopping", flush=True); return
                        time.sleep(65 if ex.code == 429 else 20 * attempt)
                    except Exception as ex:
                        print(cond, q["qid"], "attempt", attempt, "failed:", type(ex).__name__, str(ex)[:150], flush=True); time.sleep(3)
                else:
                    fh.write(json.dumps({"qid": q["qid"], "condition": cond, "model": a.model, "status": "failed_after_retries"}) + "\n"); fh.flush()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
