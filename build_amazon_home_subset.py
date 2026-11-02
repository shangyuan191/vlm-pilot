"""串流讀取 Amazon Reviews 2023 的 metadata，只保存家具、家飾、燈具相關子集。

原始檔不落地；每個來源輸出到 data/external/amazon_reviews_2023/：
  <subset>.jsonl.gz            篩選後的商品
  <counts>.json                全檔第二、三層類別統計（用來檢查篩選範圍）

用法（在專案根目錄）：
    python build_amazon_home_subset.py              # 跑全部來源
    python build_amazon_home_subset.py lighting     # 只跑指定來源
"""
import collections
import gzip
import json
import sys
import time
from pathlib import Path

from huggingface_hub import HfFileSystem

REPO_ID = 'McAuley-Lab/Amazon-Reviews-2023'
REVISION = '2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e'
OUT_DIR = Path(__file__).resolve().parent / 'data' / 'external' / 'amazon_reviews_2023'

KEEP_FIELDS = ('parent_asin', 'title', 'main_category', 'categories', 'details',
               'features', 'description', 'price', 'average_rating',
               'rating_number', 'store', 'images')

# Home & Kitchen 底下第二層屬於這些就保留；燈具關鍵字命中的也保留
HOME_SECOND_LEVEL = {'Furniture', 'Home Décor Products', 'Wall Art', 'Bedding',
                     'Bath', 'Seasonal Décor'}
HOME_KEYWORDS = ('Lamp', 'Lighting', 'Light Fixture')


def keep_home(cats):
    if len(cats) > 1 and cats[1] in HOME_SECOND_LEVEL:
        return True
    return any(k in c for c in cats for k in HOME_KEYWORDS)


def keep_lighting(cats):
    return len(cats) > 1 and cats[1] == 'Lighting & Ceiling Fans'


def keep_patio(cats):
    return len(cats) > 1 and cats[1] == 'Patio Furniture & Accessories'


def keep_handmade(cats):
    # Handmade Products > Home & Kitchen 底下，排除廚房、餐具類
    return (len(cats) > 2 and cats[1] == 'Home & Kitchen'
            and 'Kitchen' not in cats[2] and 'Dining' not in cats[2])


SOURCES = {
    'home': ('meta_Home_and_Kitchen.jsonl', keep_home,
             'home_decor_subset.jsonl.gz', 'category_counts.json'),
    'lighting': ('meta_Tools_and_Home_Improvement.jsonl', keep_lighting,
                 'lighting_subset.jsonl.gz', 'lighting_category_counts.json'),
    'patio': ('meta_Patio_Lawn_and_Garden.jsonl', keep_patio,
              'patio_furniture_subset.jsonl.gz', 'patio_category_counts.json'),
    'handmade': ('meta_Handmade_Products.jsonl', keep_handmade,
                 'handmade_home_subset.jsonl.gz', 'handmade_category_counts.json'),
}


def build(name):
    source, keep, subset_name, counts_name = SOURCES[name]
    fs = HfFileSystem()
    path = f'datasets/{REPO_ID}@{REVISION}/raw/meta_categories/{source}'
    second = collections.Counter()
    third = collections.Counter()
    kept = collections.Counter()
    n_total = n_kept = 0
    start = time.time()
    with fs.open(path, 'r', block_size=16 << 20) as src, \
            gzip.open(OUT_DIR / subset_name, 'wt', encoding='utf-8') as dst:
        for line in src:
            n_total += 1
            row = json.loads(line)
            cats = row.get('categories') or []
            second[cats[1] if len(cats) > 1 else '(none)'] += 1
            if len(cats) > 2:
                third[' > '.join(cats[1:3])] += 1
            if keep(cats):
                n_kept += 1
                kept[' > '.join(cats[1:3]) if len(cats) > 1 else '(keyword)'] += 1
                dst.write(json.dumps({k: row.get(k) for k in KEEP_FIELDS},
                                     ensure_ascii=False) + '\n')
            if n_total % 200_000 == 0:
                print(f'[{name}] {n_total:,} rows, kept {n_kept:,}, '
                      f'{time.time() - start:.0f}s', flush=True)
    summary = {
        'source': f'{REPO_ID}@{REVISION}/raw/meta_categories/{source}',
        'rows_total': n_total,
        'rows_kept': n_kept,
        'kept_by_category': kept.most_common(),
        'all_second_level': second.most_common(),
        'all_second_third_level': third.most_common(),
    }
    (OUT_DIR / counts_name).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[{name}] done: {n_total:,} rows, kept {n_kept:,}, '
          f'{time.time() - start:.0f}s', flush=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in sys.argv[1:] or SOURCES:
        build(name)


if __name__ == '__main__':
    main()
