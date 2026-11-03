# VLM product subset selection：Pilot 實驗

用 VLM 從商品候選中挑出一組商品，測兩個能力：

- **Pilot 1：Item coverage**：Subtext-Decor-90 的 20 題 × 3 個搜尋引擎 = 60 個候選池，從每池 10 個商品挑 K 個（K = 3、5）。挑出的商品要全部符合 query，並且在 query 沒提到的維度上彼此不同。
- **Pilot 2：Style alignment**：用 Amazon Reviews 2023 建 20 題有效題目。每題給一張家具圖和「lamps in this style」，從 20 盞燈挑 3 盞同風格的。題目圖有 Full / Caption / Blind 三種給法。

完整說明、prompt、結果與觀察請看實驗紀錄頁面。

## 目錄

```text
download_dataset.py                 下載 Subtext-Decor-90（固定 revision）
vlm_subset_selection.py             Pilot 1 共用：讀 parsed/queries.jsonl、圖片編碼、JSON 解析
query_mismatch_reviews.jsonl        截圖中被截斷的 query 的人工校正
query_taxonomy.json                 90 題的 query 類型標籤
build_amazon_home_subset.py         Amazon Reviews 2023：串流篩出家具／家飾／燈具
build_amazon_complete_subset.py     只留 7 個關鍵欄位都有值的商品
amazon_pool/
  build_index.py                    精簡索引 data/derived/amazon_index.jsonl.gz（295,426 個商品）
  build_pools.py                    9 題 Pilot 1 query 的 50 個候選池（雛形，未標注）
  download_images.py                下載 Pilot 2 與候選池用到的商品圖（763 張）
  pools.json, pool_stats.md
pilot1/
  build_pilot1_pool.py              匯出 20 題 × 3 引擎的候選（標注用）
  prompt.py                         選擇 prompt
  qwen_local_select.py              呼叫本機 vLLM 選 K 個
  gemini_select.py                  呼叫 Gemini（Google AI Studio）選 K 個
  score_k.py                        Oracle / Top-K / Random 期望值 / 模型的符合數與 diversity
  human_labels.json                 人工標注（591 個商品）
  {qwen3-vl-8b-bnb4,qwen3-vl-32b-bnb4,gemini-3.1-flash-lite,internvl3-8b-bnb4,llava-ov-7b-fp8}_k{3,5}_selection.jsonl   模型輸出
  internvl3-8b-bnb4-6tile_k{3,5}_selection.jsonl   InternVL3 每張圖最多 6 塊的解析度對照
pilot2/
  build_pilot2.py                   建前 20 題（seed 20261101）
  build_pilot2_extra.py             追加 7 題人工挑選的題目圖（seed 20261102）
  questions.json                    27 題（標注與模型輸出都對應這個檔案的編號）
  questions_v1_20q.json             前 20 題，用來驗證建題程式
  human_labels.json                 人工標注：題目圖 Y/N、每盞燈 P/F
  run_pilot2.py                     Caption 描述 + Full / Caption / Blind 選擇（vLLM 或 Gemini）
  score_pilot2.py                   評分
  {qwen3-vl-8b-bnb4,qwen3-vl-32b-bnb4,gemini-3.1-flash-lite,internvl3-8b-bnb4,llava-ov-7b-fp8}_{captions,full,caption,blind}.jsonl   模型輸出
  <同上>_o{1,2}_{full,caption,blind}.jsonl   換候選順序（run_pilot2.py --order-seed 1、2）
scripts/serve_qwen3vl.sh            起 Qwen3-VL-8B 的 vLLM server
scripts/serve_qwen3vl_32b.sh        起 Qwen3-VL-32B 的 vLLM server（4-bit，單卡）
scripts/serve_internvl3.sh          起 InternVL3-8B 的 vLLM server（每張圖 1 塊 448×448）
scripts/serve_llava_ov.sh           起 LLaVA-OV-7B 的 vLLM server（FP8，每張圖 384×384）
```

`data/`、`parsed/`、`models/` 不在 repo 裡（見 `.gitignore`），依下面步驟下載或放入。

## 0. 只重算分數（不需要 GPU、API key 或下載資料）

```bash
pip install -r requirements.txt
cd pilot1
python score_k.py human_labels.json 3 qwen4bit=qwen3-vl-8b-bnb4_k3_selection.jsonl lite=gemini-3.1-flash-lite_k3_selection.jsonl internvl=internvl3-8b-bnb4_k3_selection.jsonl llava=llava-ov-7b-fp8_k3_selection.jsonl qwen32b=qwen3-vl-32b-bnb4_k3_selection.jsonl
python score_k.py human_labels.json 5 qwen4bit=qwen3-vl-8b-bnb4_k5_selection.jsonl lite=gemini-3.1-flash-lite_k5_selection.jsonl internvl=internvl3-8b-bnb4_k5_selection.jsonl llava=llava-ov-7b-fp8_k5_selection.jsonl qwen32b=qwen3-vl-32b-bnb4_k5_selection.jsonl
cd ../pilot2
python score_pilot2.py --model qwen3-vl-8b-bnb4
python score_pilot2.py --model gemini-3.1-flash-lite
python score_pilot2.py --model internvl3-8b-bnb4
python score_pilot2.py --model llava-ov-7b-fp8
python score_pilot2.py --model qwen3-vl-32b-bnb4
python score_pilot2.py --model qwen3-vl-8b-bnb4_o1     # 換順序：模型名稱加 _o1、_o2
```

檢查點：Pilot 1 K = 3 應印出 `Oracle pass 2.22/3 (74%)`、`qwen4bit pass 1.75/3 (58%)`、`lite pass 1.85/3 (62%)`、`internvl pass 1.57/3 (52%)`、`llava pass 1.70/3 (57%)`、`qwen32b pass 1.88/3 (63%)`；Pilot 2 Flash-Lite 應是 Full 1.50、Caption 1.70、Blind 1.05。

## 1. Subtext-Decor-90 與 parsed 資料（Pilot 1）

已有 `parsed.zip` 的話，解壓到 repo 根目錄，會得到 `parsed/queries.jsonl` 和 `parsed/items/`。

```bash
unzip parsed.zip -d .
python pilot1/build_pilot1_pool.py      # → pilot1/web/pool.json（591 個商品）
```

原始截圖（Pilot 1 不需要）：`huggingface-cli login` 後執行 `python download_dataset.py`。

## 2. Amazon Reviews 2023（Pilot 2）

從 Hugging Face `McAuley-Lab/Amazon-Reviews-2023`（revision `2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e`）串流讀取 metadata，原始檔不存到硬碟。Home & Kitchen 約 30 分鐘、Lighting 約 11 分鐘，輸出約 1.6 GB。

```bash
python build_amazon_home_subset.py home        # → data/external/amazon_reviews_2023/home_decor_subset.jsonl.gz（1,816,552）
python build_amazon_home_subset.py lighting    # → lighting_subset.jsonl.gz（280,498）
python build_amazon_complete_subset.py         # → *_complete.jsonl.gz（402,574 + 47,368）
python amazon_pool/build_index.py              # → data/derived/amazon_index.jsonl.gz（295,426）
python amazon_pool/download_images.py          # → data/derived/amazon_images/<asin>.jpg（763 張）
```

`build_amazon_complete_subset.py` 會處理所有已建好的子集，只跑 home 和 lighting 時其他兩個會自動略過。

要拿 Amazon 資料做別的題目，直接讀索引即可：

```python
import gzip, json
rows = [json.loads(l) for l in gzip.open("data/derived/amazon_index.jsonl.gz", "rt")]
# 每列：asin, title, cats（類別路徑）, price, rating_number, store, img（大圖網址）, src,
#       以及有值時的 style / color / material / shape / finish / dims
lamps = [r for r in rows if r["cats"] and r["cats"][-1] == "Table Lamps"]
```

### 驗證建題程式（選做）

人工標注是對著附上的 `pilot2/questions.json` 做的，跑實驗請用附上的檔案。要確認建題程式能重現同一份題目：

```bash
cp pilot2/questions.json /tmp/questions.orig.json
cp pilot2/questions_v1_20q.json pilot2/questions.json
python pilot2/build_pilot2.py && python pilot2/build_pilot2_extra.py
cmp <(python -c "import json;print(json.dumps(json.load(open('/tmp/questions.orig.json')),sort_keys=True))") \
    <(python -c "import json;print(json.dumps(json.load(open('pilot2/questions.json')),sort_keys=True))") && echo same
```

## 3. 跑模型

### Qwen3-VL-8B-Instruct（vLLM 0.25.1）

```bash
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct \
  --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b --local-dir models/Qwen3-VL-8B-Instruct
scripts/serve_qwen3vl.sh 0               # 另開一個 terminal；第一個參數是 GPU 編號

cd pilot1
python qwen_local_select.py --k 3 --model qwen3-vl-8b-bnb4 --url http://127.0.0.1:8021/v1/chat/completions
python qwen_local_select.py --k 5 --model qwen3-vl-8b-bnb4 --url http://127.0.0.1:8021/v1/chat/completions
cd ..
python pilot2/run_pilot2.py --model qwen3-vl-8b-bnb4 --url http://127.0.0.1:8021/v1/chat/completions
```

輸出檔已存在時會從斷點續跑；要從頭重跑，先把對應的 `*.jsonl` 移走。

### Qwen3-VL-32B-Instruct（vLLM 0.25.1，4-bit，單卡）

```bash
huggingface-cli download Qwen/Qwen3-VL-32B-Instruct \
  --revision 0cfaf48183f594c314753d30a4c4974bc75f3ccb --local-dir models/Qwen3-VL-32B-Instruct
scripts/serve_qwen3vl_32b.sh 0          # 另開一個 terminal；port 8041，約 26 GB

cd pilot1
python qwen_local_select.py --k 3 --model qwen3-vl-32b-bnb4 --url http://127.0.0.1:8041/v1/chat/completions
python qwen_local_select.py --k 5 --model qwen3-vl-32b-bnb4 --url http://127.0.0.1:8041/v1/chat/completions
cd ..
python pilot2/run_pilot2.py --model qwen3-vl-32b-bnb4 --url http://127.0.0.1:8041/v1/chat/completions
```

不要用 `--tensor-parallel-size 2` 分到兩張卡：vLLM 0.25.1 的 bitsandbytes 4-bit 這樣跑會輸出亂碼。

### InternVL3-8B-Instruct（vLLM 0.25.1）

```bash
huggingface-cli download OpenGVLab/InternVL3-8B-Instruct \
  --revision ddb3a169d5582e5c76e0809a128e55ab63686ada --local-dir models/InternVL3-8B-Instruct
scripts/serve_internvl3.sh 0            # 另開一個 terminal；port 8031

cd pilot1
python qwen_local_select.py --k 3 --model internvl3-8b-bnb4 --url http://127.0.0.1:8031/v1/chat/completions
python qwen_local_select.py --k 5 --model internvl3-8b-bnb4 --url http://127.0.0.1:8031/v1/chat/completions
cd ..
python pilot2/run_pilot2.py --model internvl3-8b-bnb4 --url http://127.0.0.1:8031/v1/chat/completions
```

每張圖限制成 1 塊 448×448：`serve_internvl3.sh` 會建 `models/InternVL3-8B-Instruct-1tile`，只改 `config.json` 的 `max_dynamic_patch`（12 → 1），權重用連結指回原檔。InternVL3 回傳的 JSON 有時鍵沒加引號（`{4: 1}`），`vlm_subset_selection.parse_json` 會補上引號再解析。

### LLaVA-OneVision-Qwen2-7B（vLLM 0.25.1，FP8）

```bash
huggingface-cli download llava-hf/llava-onevision-qwen2-7b-ov-hf \
  --revision 0d50680527681998e456c7b78950205bedd8a068 --local-dir models/llava-onevision-qwen2-7b-ov-hf
scripts/serve_llava_ov.sh 0             # 另開一個 terminal；port 8032

cd pilot1
python qwen_local_select.py --k 3 --model llava-ov-7b-fp8 --url http://127.0.0.1:8032/v1/chat/completions
python qwen_local_select.py --k 5 --model llava-ov-7b-fp8 --url http://127.0.0.1:8032/v1/chat/completions
cd ..
python pilot2/run_pilot2.py --model llava-ov-7b-fp8 --url http://127.0.0.1:8032/v1/chat/completions
```

vLLM 0.25.1 不支援這個模型的 bitsandbytes 4-bit，所以用 FP8。`serve_llava_ov.sh` 會建 `models/llava-onevision-qwen2-7b-ov-hf-384`，把 `image_grid_pinpoints` 改成 `[[384, 384]]`，讓 Pilot 2 的 21 張圖放得進 32k 上下文。LLaVA-OV 偶爾在 `selection_notes` 裡寫出沒跳脫的引號；`run_pilot2.py` 遇到整段 JSON 無法解析時只抓出 `selected_positions`。

### 換候選順序（Pilot 2）

Pilot 2 的主要結果是 3 種候選順序的平均：建題時的順序，加上 `--order-seed 1`、`--order-seed 2`。

```bash
python pilot2/run_pilot2.py --model <served-name> --url <url> --order-seed 1
python pilot2/run_pilot2.py --model <served-name> --url <url> --order-seed 2
```

順序由 `random.Random(f"{seed}-{qid}")` 決定。輸出的 `selected_positions` 已換回原本編號（`score_pilot2.py` 直接可用），`shown_positions` 是模型實際看到的編號。Caption 條件沿用 `<model>_captions.jsonl`。

### InternVL3 解析度對照（Pilot 1）

把 `serve_internvl3.sh` 建的資料夾改成 `max_dynamic_patch = 6`（另存 `models/InternVL3-8B-Instruct-6tile`），以 `internvl3-8b-bnb4-6tile` 名稱服務，`--gpu-memory-utilization 0.50 --max-model-len 22528 --limit-mm-per-prompt '{"image": 10}'`，再跑 `qwen_local_select.py`。結果：K = 3 是 1.57/3（和 1 塊相同），K = 5 是 2.53/5（1 塊是 2.63）。

### Gemini 3.1 Flash-Lite（Google AI Studio 免費 API）

API key 存在 `~/.config/gemini_api_key`，程式從這個檔案讀，不要寫進程式或指令：

```bash
mkdir -p ~/.config
read -rs "KEY?Paste API key: " && printf '%s' "$KEY" > ~/.config/gemini_api_key && unset KEY   # zsh
# bash：read -rsp "Paste API key: " KEY && printf '%s' "$KEY" > ~/.config/gemini_api_key && unset KEY
chmod 600 ~/.config/gemini_api_key

cd pilot1
python gemini_select.py --k 3 --model gemini-3.1-flash-lite
python gemini_select.py --k 5 --model gemini-3.1-flash-lite
cd ..
python pilot2/run_pilot2.py --backend gemini --model gemini-3.1-flash-lite
```

## 資料使用

Amazon Reviews 2023 沒有明確標示授權，這裡只用於研究。repo 不包含完整的 Amazon metadata 或任何商品圖；只有題目用到的少量商品（ASIN、標題、圖片網址），完整資料與圖片請依上面步驟自行下載。

Hou et al., *Bridging Language and Items for Retrieval and Recommendation*, arXiv:2403.03952, 2024.
