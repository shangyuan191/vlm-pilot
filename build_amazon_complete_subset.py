"""從 build_amazon_home_subset.py 的子集中，只保留判斷所需欄位都有資料的商品。

必填欄位：title、categories、images、details、features、description、price。
（main_category、store 與判斷 query 是否相符無關，不列為必填。）
原子集不動，另存為 <subset>_complete.jsonl.gz，統計寫入 complete_counts.json。

用法（在專案根目錄）：
    python build_amazon_complete_subset.py
"""
import collections
import gzip
import json

from build_amazon_home_subset import OUT_DIR, SOURCES

REQUIRED = ('title', 'categories', 'images', 'details', 'features',
            'description', 'price')


def filled(value):
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(filled(v) for v in value)
    return value not in (None, {})


def main():
    summary = {'required_fields': list(REQUIRED), 'subsets': {}}
    for name, (_, _, subset_name, counts_name) in SOURCES.items():
        subset, counts = OUT_DIR / subset_name, OUT_DIR / counts_name
        if not (counts.exists() and counts.stat().st_mtime >= subset.stat().st_mtime):
            print(f'略過尚未篩選完成的 {subset_name}')
            continue
        out = OUT_DIR / subset_name.replace('_subset.jsonl.gz', '_complete.jsonl.gz')
        n = kept = 0
        kept_by = collections.Counter()
        total_by = collections.Counter()
        with gzip.open(subset, 'rt', encoding='utf-8') as src, \
                gzip.open(out, 'wt', encoding='utf-8') as dst:
            for line in src:
                n += 1
                row = json.loads(line)
                cats = row.get('categories') or []
                group = ' > '.join(cats[:2])
                total_by[group] += 1
                if all(filled(row.get(k)) for k in REQUIRED):
                    kept += 1
                    kept_by[group] += 1
                    dst.write(line)
        summary['subsets'][name] = {
            'source': subset_name, 'output': out.name,
            'rows_total': n, 'rows_kept': kept,
            'kept_by_category': [(g, kept_by[g], total_by[g])
                                 for g, _ in total_by.most_common()],
        }
        print(f'[{name}] {kept:,}/{n:,} ({kept / n:.1%}) -> {out.name}', flush=True)
    (OUT_DIR / 'complete_counts.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
