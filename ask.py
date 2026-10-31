#!/usr/bin/env python3
"""Ask my blog - retrieval-augmented answers over everything I've written, with citations.

    python ask.py "Why do propensity models mislead campaign targeting?"
    python ask.py --show "what does last-click get wrong"     # print the retrieved passages

Retrieval is TF-IDF + cosine over ~900-character chunks (index.json, built by build_index.py).
Generation is optional: with GEMINI_API_KEY or ANTHROPIC_API_KEY set, the model answers
using ONLY the retrieved passages and cites them as [1], [2]...  Without a key, you get the
passages themselves - which is often enough, and is also the honest baseline every RAG system
should be compared against.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent))
import llm  # noqa: E402

INDEX = Path(__file__).parent / "index.json"
TOP_K = 5

SYSTEM = """You answer questions about Nikunj Kaushik's writing on analytics, data science and AI.
Use ONLY the numbered passages provided. Cite every claim with [n]. If the passages don't contain
the answer, say so in one sentence - do not fill the gap from general knowledge.
Write in plain English, 3-6 sentences, no preamble."""


def load():
    if not INDEX.exists():
        raise SystemExit("index.json missing - run: python build_index.py")
    return json.loads(INDEX.read_text(encoding="utf-8"))


def retrieve(docs, question: str, k: int = TOP_K):
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", min_df=1)
    M = vec.fit_transform([d["text"] for d in docs])
    q = vec.transform([question])
    sims = cosine_similarity(q, M).ravel()
    order = sims.argsort()[::-1]
    out, seen_pages = [], {}
    for i in order:
        d = docs[i]
        if seen_pages.get(d["url"], 0) >= 2:        # at most two chunks per page
            continue
        seen_pages[d["url"]] = seen_pages.get(d["url"], 0) + 1
        out.append((float(sims[i]), d))
        if len(out) == k:
            break
    return out


def answer(question: str, hits) -> str | None:
    passages = "\n\n".join(f"[{n}] ({d['title']}) {d['text']}" for n, (_, d) in enumerate(hits, 1))
    text = llm.complete(SYSTEM, f"Question: {question}\n\nPassages:\n{passages}", max_tokens=450)
    if text and not re.search(r"\[\d+\]", text):
        return None  # no citations = not grounded; show passages instead
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--show", action="store_true", help="print retrieved passages")
    ap.add_argument("-k", type=int, default=TOP_K)
    a = ap.parse_args()

    docs = load()
    hits = retrieve(docs, a.question, a.k)
    print(f"index: {len(docs)} chunks · provider: {llm.provider()}\n")

    ans = answer(a.question, hits) if llm.provider() != "none" else None
    if ans:
        print(ans, "\n")
    else:
        print("No model configured (or answer not grounded) - here is what the retriever found:\n")
    if a.show or not ans:
        for n, (s, d) in enumerate(hits, 1):
            snippet = re.sub(r"\s+", " ", d["text"])[:420]
            print(f"[{n}] {d['title']}  (score {s:.2f})\n    {snippet}…\n")
    print("Sources:")
    for n, (_, d) in enumerate(hits, 1):
        print(f"  [{n}] {d['title']} - {d['url']}")


if __name__ == "__main__":
    main()
