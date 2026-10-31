#!/usr/bin/env python3
"""Build the retrieval index from the live site.

Reads https://nikunjkaushik.com/llms.txt for the article list (the site publishes a
machine-readable profile exactly so tools like this don't have to scrape navigation),
fetches each article, strips it to text, chunks it, and writes index.json.

    python build_index.py            # ~25 pages, a few seconds
"""
from __future__ import annotations

import html
import json
import re
import urllib.request
from pathlib import Path

SITE = "https://nikunjkaushik.com"
OUT = Path(__file__).parent / "index.json"
CHUNK, OVERLAP = 900, 150


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "ask-my-blog/1.0 (+github.com/kaushiknikunj1)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", errors="replace")


def article_urls() -> list[tuple[str, str]]:
    txt = fetch(f"{SITE}/llms.txt")
    seen, out = set(), []
    for m in re.finditer(r"\[([^\]]+)\]\((https://nikunjkaushik\.com/(?:blog|projects)/[^)]+\.html)\)", txt):
        title, url = m.group(1), m.group(2)
        if url not in seen:
            seen.add(url); out.append((title, url))
    return out


def to_text(page: str) -> str:
    page = re.sub(r"<(script|style|nav|footer|svg)[^>]*>.*?</\1>", " ", page, flags=re.S | re.I)
    page = re.sub(r"<br\s*/?>|</(p|h\d|li|div|tr)>", "\n", page, flags=re.I)
    text = html.unescape(re.sub(r"<[^>]+>", " ", page))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    # drop the shared header lines
    text = re.sub(r"^.*?(Back to (Blog|Projects)|Home Projects Blog AI Tools Contact)\s*", "", text, count=1, flags=re.S)
    return text.strip()


def chunks(text: str) -> list[str]:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    out, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 1 > CHUNK and cur:
            out.append(cur)
            tail = cur[-OVERLAP:]
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail   # start overlap on a word boundary
            cur = (tail + " " + p) if OVERLAP else p
        else:
            cur = (cur + "\n" + p).strip()
    if cur:
        out.append(cur)
    return [c for c in out if len(c) > 120]


def main():
    docs = []
    for title, url in article_urls():
        try:
            text = to_text(fetch(url))
        except Exception as e:  # noqa: BLE001
            print(f"skip {url}: {e}"); continue
        cs = chunks(text)
        docs += [{"title": title, "url": url, "chunk": i, "text": c} for i, c in enumerate(cs)]
        print(f"{len(cs):3d} chunks  {title}")
    OUT.write_text(json.dumps(docs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT.name}: {len(docs)} chunks from {len({d['url'] for d in docs})} pages")


if __name__ == "__main__":
    main()
