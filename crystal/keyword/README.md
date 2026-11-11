# Deceptive Keyword Identification (crystal/keyword)

Implementation of Section 2.3 of the paper, *Deceptive Keyword Identification and Explanation* (the `Deceptive-Keyword Extraction` block in Figure 1). It produces the **deceptive keywords** and **sentences** highlighted on the article view (B-2), following the paper's five steps.

| Paper step (Sec. 2.3 / Figure 1) | Implementation | File |
| --- | --- | --- |
| ① Generate veracity label | 10-fold RoBERTa verdict (this model is the verdict) | `crystal/veracity/` |
| ② Token attribution — IG/LIME Attribution | Token attribution toward the predicted label | `ig.py`, `lime_words.py` |
| ③ Cue Filtering | Keep only highly attributed tokens explainable as **deception cues**, collected into a cue lexicon; **person names excluded** | `lexicon.py` |
| ④ Sentence Attribution | Rank verdict-contributing sentences via ContextCite-style ablation | `contextcite.py` |
| ⑤ LLM explanation | Keyword/article rationales, plus claim contrast when an opposing-veracity article exists | `crystal/rationale/` |

The final highlight file is assembled by `build_highlights.py`, which merges the top sentences (④) and cue words (③) into `cc_keywords.jsonl`; the web app applies it when `CRYSTAL_KEYWORDS=cc`.

```
articles.jsonl ─┬─ mask_names.py ──► masked.csv ──► roberta_veracity/train_masked.py ──► fold0_masked/best
                │        (auxiliary model that removes person-name influence during ② attribution)        │
                └──────────────────────────── contextcite.py (④ sentence attribution, top 5) ◄────────────┘
                                                      │
                                             cc_sentences_top5.jsonl
                                                      │
                                   build_highlights.py  +  lexicon.py (③ cue lexicon)
                                                      │
                                      cc_keywords.jsonl  (+ rationale cache)
                                                      │
                                run the web server with CRYSTAL_KEYWORDS=cc → article highlighting
```

## ② Token Attribution (IG/LIME Attribution)

`ig.py` (Integrated Gradients + spaCy phrase expansion; the `CRYSTAL_KEYWORDS` default `ig`) and `lime_words.py` (top-K LIME word attribution) compute each token's attribution toward the predicted label. These attributions reveal which words and phrases drive the verdict, and they are the basis for building the cue lexicon in ③.

**Person-name masking (paper: *person names are excluded*).** The original model uses certain person names (e.g., Bieber, Gomez) as strong signals regardless of veracity. To remove this, the attribution model is trained separately on name-masked text — `mask_names.py` uses spaCy to replace person entities (PERSON) with `[NAME]` and writes a CSV, and `../../roberta_veracity/train_masked.py` trains the `fold0_masked` model on that text (for attribution / sentence ablation only, test acc 0.8489). **The verdict itself still comes from the original 10-fold model.**

## ③ Cue Filtering — `lexicon.py`

Among the tokens attributed highly in ②, only those **explainable as deception cues** are kept and collected into a cue lexicon. **Person names**, which score high regardless of veracity, are excluded.

- **FAKE list**: anonymous sourcing, hedging, exaggeration, profanity, emotion words
- **TRUE list**: named statements, spokespersons, confirmation, official records, first-person posts, interviews, dates, news outlets

Each entry is a (regex, explanation template) pair. During highlighting, only the list matching the **predicted label** is matched against the body, and the word explanations are written to the cache from the templates.

## ④ Sentence Attribution — `contextcite.py`

A ContextCite-style method adapted to a classifier, implemented directly. It fits a LASSO linear surrogate from the verdict log-odds of 256 inputs with sentences randomly ablated, obtains per-sentence contributions, and stores the top 5 in original-text coordinates. Because a few words rarely suffice to explain a verdict (paper Sec. 2.3), sentence-level evidence is highlighted as well.

## ⑤ Assembly — `build_highlights.py`

Merges sentences (1–5 depending on body length; replaced by the next-ranked one if adjacent/overlapping) and cue words (only the predicted-label list, excluding words inside highlighted sentences, up to 15) into `cc_keywords.jsonl`. Word explanations go to the cache from the ③ templates; sentence- and article-level explanations are produced by the ⑤ LLM (`crystal/rationale/`).

## Run order

Run everything from the project root (on Windows PowerShell, prefix with `$env:KMP_DUPLICATE_LIB_OK='TRUE'`).

```
python -X utf8 crystal/keyword/mask_names.py
python -X utf8 roberta_veracity/train_masked.py
python -X utf8 crystal/keyword/contextcite.py --model roberta_veracity/folds/fold0_masked/best --ids ids.json --masked crystal/data/keyword/masked.csv
python -X utf8 crystal/keyword/build_highlights.py
$env:CRYSTAL_KEYWORDS='cc'; python -X utf8 -m uvicorn crystal.web.app:app --port 8765
```

`ids.json` is a list of target article_ids (a JSON array). For the demo we used the two demo users' 10 feed pages plus 393 related articles.

## Output format (`cc_keywords.jsonl`, one line = one article)

```json
{"article_id": "gossipcop-7455413235", "final_label": 1, "method": "contextcite+lexicon",
 "keywords": [{"phrase": "rumours", "field": "body", "start": 200, "end": 207, "score": 0.6, "sentence": "...", "kind": "lexicon"},
              {"phrase": "(sic),’ wrote one fan ...", "field": "body", "start": 1756, "end": 1839, "score": 1.0, "sentence": "...", "kind": "sentence"}]}
```

`kind` `sentence` is a sentence highlight (④), `lexicon` is a cue-word highlight (③). Explanations are stored in `crystal/data/rationale_cache_lime_masked.jsonl` keyed by `(article_id, kind=keyword, key=phrase)`; if absent, the LLM generates them in the web app (⑤).
