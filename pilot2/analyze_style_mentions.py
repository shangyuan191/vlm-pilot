"""Pilot 2 analysis (2026-11-04, no model calls): does a text name the reference furniture's true style, and do
answers whose text names it score higher? Keyword match (Chinese + English) for the 5 styles over: each caption
source, Full selection_notes and fulldesc reference_description; plus the "same colour, other style" trap rate.
Keyword matching is a rough proxy. Usage: python analyze_style_mentions.py
"""
import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))
import json, re, statistics as st
from pathlib import Path
KW={"mid-century":r"中世紀|世紀中|mid[- ]?century|mcm","farmhouse":r"農舍|農莊|鄉村|田園|farmhouse|rustic","industrial":r"工業|industrial|loft","coastal":r"海岸|海洋|濱海|海灘|海邊|沿海|coastal|nautical|beach","scandinavian":r"北歐|斯堪|scandinavian|nordic"}
def styles_in(t): return {s for s,p in KW.items() if re.search(p,t,re.I)}
def cat(t,gold):
    s=styles_in(t)
    if not s: return "none"
    if s=={gold}: return "correct"
    if gold in s: return "mixed"
    return "wrong"
labels=json.load(open('human_labels.json'))
Q={q["qid"]:q for q in json.load(open('questions.json'))}
scored=[q for q in Q if labels[q]["query"]!="N"]
lab=lambda q,p: labels[q]["items"][str(p)]
def per_q(tag,cond):
    out={}
    for s in ("","_o1","_o2"):
        f=Path(f"{tag}{s}_{cond}.jsonl")
        for r in map(json.loads,f.read_text().splitlines()):
            if r.get("status")=="ok" and r["qid"] in scored:
                out.setdefault(r["qid"],[]).append((sum(lab(r["qid"],p)=="P" for p in r["selected_positions"]),r))
    return out
models=["qwen3-vl-32b-bnb4","gemini-3.1-flash-lite","qwen3-vl-8b-bnb4","internvl3-8b-bnb4","llava-ov-7b-fp8"]
caps={m:{r["qid"]:r["caption"] for r in map(json.loads,open(f"{m}_captions.jsonl")) if r.get("status")=="ok"} for m in models+["claude-opus-5-5"]}
print("== caption style naming (20 scored qids) ==")
capcat={}
for m,c in caps.items():
    capcat[m]={q:cat(c.get(q,""),Q[q]["style"]) for q in scored}
    from collections import Counter; print(m, dict(Counter(capcat[m].values())))
    for q in scored:
        if capcat[m][q] in("wrong","mixed"): print("   ",q,capcat[m][q],styles_in(caps[m][q]))
def split(tag,cond,cc):
    pq=per_q(tag,cond); g={}
    for q,v in pq.items(): g.setdefault(cc[q],[]).append(st.mean(x[0] for x in v))
    return {k:f"{st.mean(v):.2f} (n={len(v)})" for k,v in sorted(g.items())}
print("\n== Caption score split by whether caption names the correct style ==")
for m in models:
    print(m,"own:",split(m,"caption",capcat[m]))
    print(m,"claude:",split(f"{m}_cap-claude-opus-5-5","caption",capcat["claude-opus-5-5"]))
    if m!="qwen3-vl-32b-bnb4": print(m,"q32b:",split(f"{m}_cap-qwen3-vl-32b-bnb4","caption",capcat["qwen3-vl-32b-bnb4"]))
    print(m,"stylename same qids split by claude-cap category:",split(m,"stylename",capcat["claude-opus-5-5"]))
print("\n== Full / fulldesc: does the model's own text name the correct style? ==")
for m in models:
    for cond,field in (("full","selection_notes"),("fulldesc","reference_description")):
        pq=per_q(m,cond); g={}
        for q,v in pq.items():
            for sc,r in v: g.setdefault(cat(r.get(field) or "",Q[q]["style"]),[]).append(sc)
        print(m,cond,field,{k:f"{st.mean(v):.2f} (n={len(v)})" for k,v in sorted(g.items())})
print("\n== near-miss trap rate (picked same-colour lamp labelled not same style) ==")
bucket=lambda q,p: next(c["bucket"] for c in Q[q]["candidates"] if c["pos"]==p)
for m in models:
    row=[]
    for cond in ("full","caption","stylename"):
        pq=per_q(m,cond); t=[sum(bucket(q,p)=="near-miss" and lab(q,p)!="P" for p in r["selected_positions"]) for v in pq.values() for _,r in v]
        row.append(f"{cond} {sum(t)/(3*len(t)):.0%}")
    print(m,row)
# base rate of near-miss-not-P among candidates
nm=[sum(c["bucket"]=="near-miss" and lab(q,c["pos"])!="P" for c in Q[q]["candidates"])/len(Q[q]["candidates"]) for q in scored]
print("random trap rate",f"{st.mean(nm):.0%}")
