"""Pilot 1 selection prompt (pick exactly K of up to 10 candidates)."""

QUERY_IDS = [1, 8, 11, 13, 26, 28, 30, 33, 38, 41, 43, 44, 49, 50, 71, 72, 74, 75, 76, 86]
PROMPT_VERSION = 2


def make_prompt(k, query, engine, n):
    return (f"你是商品搜尋結果的 subset selection 評審。QUERY: {query}\nENGINE: {engine}\n"
            f"以下有 {n} 個依排名排列的候選商品。請剛好選出 {k} 個，必須同時滿足："
            "(1) 每一個都盡量符合 query 的所有條件；(2) 彼此在外觀、材質、顏色、形狀或風格上盡量多樣。"
            f"即使符合的商品不足 {k} 個，也必須選滿 {k} 個（選最接近的）。"
            "請只回傳 JSON，不要 Markdown。格式如下（尖括號是要你填入的內容，不是答案）："
            '{"selected_positions":[<position>, ...],"relevance_by_position":{"<position>":<1 符合 / 0 不符合>, ...},"selection_notes":"<一句話說明為什麼這樣選>"}。'
            f"selected_positions 使用 1-based position，剛好 {k} 個且不得重複；請根據圖片和標題實際判斷後再選。")
