#!/usr/bin/env python3
"""コーパスを「発音」のかな列に変換する — DRPNN に音として聞かせるための前処理 (2026-08-27)

## なぜ前処理なのか

**形態素解析器は脳の一部ではない。** 人が文字を読んで声に出すのと同じで、
漢字→かな変換は DRPNN の外側の作業である。DRPNN 本体には入れない。

## 読み(kana) ではなく 発音(pron) を使う

音として聞かせるので、**表記ではなく発音**が要る。

| 原文 | 読み(kana) | **発音(pron)** |
|---|---|---|
| 今日は良い天気ですね | キョウ**ハ**ヨイテンキデスネ | キョー**ワ**ヨイテンキデスネ |
| こんにちは、東京へ行きます | コンニチ**ハ**、トウキョウ**ヘ**イキマス | コンニチ**ワ**、トーキョー**エ**イキマス |
| 彼女は学校を出た | カノジョ**ハ**ガッコウ**ヲ**デタ | カノジョ**ワ**ガッコー**オ**デタ |

助詞の は→ワ / へ→エ / を→オ が正しく変わり、長音が ウ ではなく ー になる。
ー は `kana.rs` の `Mora::Long` (直前の母音を伸ばす) に正しく乗る。

## 決めたこと (実行前に宣言)

1. **発音 (pron) を使う。** 読み (kana) は使わない。
2. **pron が無いトークン (記号・絵文字・英字など) は落とす。** 音にならないため。
3. **カタカナのまま出す。** `moras_from_kana` が `to_hiragana` で処理する。
4. **1 投稿 = 1 行。** 投稿の間に区切りは入れない。
   *実際の発話には間 (ポーズ) があり、それが適応をリセットするが、
   いまのモーラ表に「無音」を表す要素が無い (っ は促音であってポーズではない)。
   **既知の穴として記録し、ここでは入れない。***
5. **本文は一切印字しない。** 数値のみ。コーパスは実機に留まる。

## 出力

`data/corpus/roleplay_kana.txt` — `.gitignore:27` の `data/corpus/` 配下なので追跡されない。

使い方: python python/corpus_to_kana.py [入力 jsonl] [出力 txt] [スレッド数上限]
"""
import json
import sys
import time
import collections

def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "data/corpus/roleplay_filtered.jsonl"
    dst = sys.argv[2] if len(sys.argv) > 2 else "data/corpus/roleplay_kana.txt"
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0  # 0 = 全部

    import fugashi
    tagger = fugashi.Tagger()

    print("=== コーパスを発音のかな列に変換 ===")
    print()
    print("【方針】形態素解析器は脳の一部ではない。これは前処理であり DRPNN 本体には入れない。")
    print("【選択】読み(kana) ではなく **発音(pron)** を使う (助詞の は→ワ・へ→エ・を→オ)。")
    print("【原則】本文は一切印字しない。数値のみ。コーパスは実機に留まる。")
    print()
    print(f"入力: {src}")
    print(f"出力: {dst}  (.gitignore の data/corpus/ 配下 = 追跡されない)")
    print(f"上限: {'全部' if limit == 0 else f'{limit} スレッド'}")
    print()

    t0 = time.time()
    threads = posts = 0
    src_chars = out_chars = 0
    dropped_tokens = kept_tokens = 0
    dropped_chars = 0
    char_kinds = collections.Counter()

    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as g:
        for line in f:
            if limit and threads >= limit:
                break
            threads += 1
            d = json.loads(line)
            texts = [d.get("first_post", "")] + [
                q.get("post_content", "") for q in d.get("posts", [])
            ]
            for t in texts:
                t = t.replace("\n", " ").replace("\r", " ").strip()
                if not t:
                    continue
                posts += 1
                src_chars += len(t)
                out = []
                for w in tagger(t):
                    pron = getattr(w.feature, "pron", None)
                    if pron and pron != "*":
                        out.append(pron)
                        kept_tokens += 1
                    else:
                        dropped_tokens += 1
                        dropped_chars += len(w.surface)
                s = "".join(out)
                if s:
                    out_chars += len(s)
                    for ch in s:
                        o = ord(ch)
                        if 0x30A1 <= o <= 0x30F6:
                            char_kinds["カタカナ"] += 1
                        elif ch == "ー":
                            char_kinds["長音符"] += 1
                        elif 0x3041 <= o <= 0x3096:
                            char_kinds["ひらがな"] += 1
                        else:
                            char_kinds["その他"] += 1
                    g.write(s + "\n")

    el = time.time() - t0
    print(f"--- 結果 ---")
    print(f"  スレッド            : {threads:,}")
    print(f"  投稿                : {posts:,}")
    print(f"  元の文字数          : {src_chars:,}")
    print(f"  **変換後の文字数**  : {out_chars:,}  (元の {out_chars/max(src_chars,1)*100:.1f}%)")
    print(f"  採用したトークン    : {kept_tokens:,}")
    print(f"  落としたトークン    : {dropped_tokens:,}  ({dropped_tokens/max(kept_tokens+dropped_tokens,1)*100:.1f}%)")
    print(f"  落とした文字        : {dropped_chars:,}  (元の {dropped_chars/max(src_chars,1)*100:.1f}%)")
    print()
    print(f"  変換後の文字構成:")
    tot = sum(char_kinds.values())
    for k, v in char_kinds.most_common():
        print(f"    {k:<8} {v:>12,}  {v/max(tot,1)*100:>5.1f}%")
    print()
    print(f"  所要 {el:.1f} 秒  ({src_chars/max(el,1e-9)/1e6:.2f} M文字/秒)")
    print()
    print("  【既知の穴】投稿の間に区切り (ポーズ) を入れていない。")
    print("  実際の発話には間があり、それが適応をリセットするが、いまのモーラ表に")
    print("  「無音」を表す要素が無い (っ は促音であってポーズではない)。")

if __name__ == "__main__":
    main()
