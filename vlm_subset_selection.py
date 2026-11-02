"""Shared helpers for Pilot 1: read the parsed Subtext-Decor-90 queries and encode product crops.

Expects the parsed data at ./parsed/queries.jsonl and ./parsed/items/ (parsed.zip unzipped at the repo root).
Queries truncated in the screenshots are replaced by the reviewed text in query_mismatch_reviews.jsonl.
"""
import base64, io, json, re
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent
PARSED = ROOT / "parsed" / "queries.jsonl"
REVIEWS = ROOT / "query_mismatch_reviews.jsonl"
ENGINES = ("amazon", "google_shopping", "onton")


def img_url(path):
    """Downscale to at most 900x900 and return a base64 JPEG data URL."""
    with Image.open(path) as im:
        im.thumbnail((900, 900)); b = io.BytesIO(); im.convert("RGB").save(b, format="JPEG", quality=80, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def parse_json(t):
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t.strip(), flags=re.I)
    try: return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, flags=re.S)
        if not m: raise
        try: return json.loads(m.group(0))
        except json.JSONDecodeError:
            # some models (e.g. InternVL3) write bare integer keys: {4: 1}; quote them and retry
            return json.loads(re.sub(r'([{,]\s*)(\d+)\s*:', r'\1"\2":', m.group(0)))


def rows():
    out = {str(x["query_id"]): x for x in map(json.loads, PARSED.read_text().splitlines())}
    for line in REVIEWS.read_text().splitlines():
        if line.strip():
            r = json.loads(line); q = str(r["query_id"])
            if q in out: out[q]["query"] = r["canonical_query"]
    return out


def items(row, engine):
    """Top-10 candidates of one engine, in ranking order."""
    return list(row["engines"][engine].get("items", []))[:10]
