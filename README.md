# Ask my blog

Retrieval-augmented answers over everything I've written — 27 articles and case studies, ~520 passages — with citations back to the source. Built the way I argue RAG for analytics should be built: the retriever is the product, the generator is optional, and the baseline you compare against is "just show the passages".

```
$ python ask.py "Why do propensity models mislead campaign targeting?"

index: 519 chunks · provider: none

[1] Uplift Modeling for Incremental Targeting  (score 0.23)
    ...the right question isn't "who is likely to convert?" — it's "whose probability of
    converting goes up because of the campaign?" Propensity models can't tell you that...
[2] Uplift Modeling for Incremental Targeting  (score 0.21)
[3] Uplift Modeling: Measuring the True Impact of Your Campaigns  (score 0.16)
...
Sources:
  [1] Uplift Modeling for Incremental Targeting - https://nikunjkaushik.com/projects/uplift-modeling.html
  [3] Uplift Modeling: Measuring the True Impact of Your Campaigns - https://nikunjkaushik.com/blog/uplift-modeling.html
```

That is the no-key baseline. With a key, a cited 3–6 sentence answer is written from exactly those passages.

## How it works

- **`build_index.py`** reads the article list from [`/llms.txt`](https://nikunjkaushik.com/llms.txt) — the site publishes a machine-readable profile so tools don't have to scrape navigation — fetches each page, strips it to text, chunks at ~900 characters with word-boundary overlap, and writes `index.json`.
- **`ask.py`** retrieves with TF-IDF (1–2 grams, sublinear tf) + cosine, caps at two chunks per page so one long article can't crowd out the rest, and prints the top five.
- With `GEMINI_API_KEY` or `ANTHROPIC_API_KEY` set, the model answers **using only the retrieved passages** and must cite `[n]`. An answer with no citations is treated as ungrounded and the passages are shown instead.
- Without a key, you get the passages. That's the honest baseline.

## Run it

```bash
pip install -r requirements.txt
python build_index.py                     # refresh from the live site (optional; index.json is committed)
python ask.py "what does last-click attribution get wrong?"
python ask.py --show "how should I pick a north star metric"
```

## Why TF-IDF and not embeddings

For 500 passages written by one person on a narrow set of topics, lexical retrieval is competitive and fully inspectable. When a result is wrong I can see exactly which terms matched. Swapping in embeddings is a ten-line change; I'd do it when the corpus or the vocabulary gets wide enough that synonyms start to matter.

Related writing: [RAG for analytics](https://nikunjkaushik.com/blog/rag-for-analytics.html).
