# CRYSTAL: A Credibility-aware News Recommendation System for Explainable Fake News Mitigation

Code for our **WWW Companion '27** paper (The ACM Web Conference 2027, Demo Track).
Personalized news recommendation is agnostic to whether articles are true, so users consume fake news without noticing it. CRYSTAL is an interactive web system that mitigates this by **predicting and explaining veracity and letting users explore opposing-veracity coverage**. Across three levels — feed, article, and event — it surfaces veracity so that fake news is kept *visible, explained, and correctable* rather than silently filtered out.

- Live demo: https://crystal-1-2wtv.onrender.com/
- Demo video: https://youtu.be/CTDt7pn6cRk

---

## Project Structure

```
crystal-www27/
├── crystal/
│   ├── data/                 # Pipeline outputs & weights (not included — see below)
│   ├── common.py             # Shared utilities (is_displayable, etc.)
│   ├── prepare_data.py       # Source CSV → articles.jsonl / users.jsonl
│   ├── veracity/             # RoBERTa veracity predictor (load, predict, temperature scaling)
│   ├── keyword/              # Deceptive keyword & sentence highlighting pipeline
│   ├── event/                # Event clustering & opposing-veracity matching
│   ├── recsys/               # CROWN recommendation, feed generation, veracity distribution
│   ├── rationale/            # LLM (Gemini) rationale generation & cache
│   └── web/                  # FastAPI web app (feed / article / compare views)
├── roberta_veracity/         # RoBERTa training (train / train_kfold / train_masked)
├── crown/                    # CROWN checkpoint & source (not included)
├── requirements.txt
└── Dockerfile
```

---

## Setup

```bash
pip install -r requirements.txt          # Minimal deps to run the web app (fastapi, uvicorn, google-genai)
```

To run the full pipeline (preprocessing, veracity, keywords, events, recommendation), install the additional dependencies:

```bash
pip install torch transformers sentence-transformers hdbscan spacy captum \
            pandas numpy scikit-learn torch-geometric
python -m spacy download en_core_web_sm   # For person-name masking in the keyword pipeline
```

GPU (recommended for veracity / IG / ContextCite):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Data

CRYSTAL uses the **GossipCop** benchmark from FakeNewsNet — entertainment news labeled true/fake by fact-checkers. We use the **8.0k** articles with bodies of at least 150 words for recommendation and display; the Twitter engagements of **396k** users over these articles provide the reading histories of the demo users.

Data files are not included in the repository. Obtain the source data from [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet) and preprocess it, or use the demo image that bundles the pipeline outputs (see Docker below).

| File | Description |
|---|---|
| `crystal/data/articles.jsonl` | Articles (title, body, label, veracity prediction, keywords, counterparts) |
| `crystal/data/users.jsonl` | User reading histories |
| `crystal/data/feeds.jsonl` | Pre-computed recommendation feed per user |

---

## Models

| Model | Source | Role |
|---|---|---|
| RoBERTa | `roberta-base` fine-tuned (`roberta_veracity/`) | Veracity prediction (10-fold out-of-fold + temperature scaling) |
| CROWN | [seongeunryu/crown-www25](https://github.com/seongeunryu/crown-www25) | Personalized recommendation |

Download the trained models from [Google Drive](https://drive.google.com/drive/folders/1dOpC3oqhkjTOJ7uKgsDZmscVZf4ofvYp?usp=sharing).
The final layout should be:

```
roberta_veracity/
└── best/                 # Veracity model (config.json + model.safetensors + tokenizer)

crown/
├── crown-www25/          # CROWN source code (model.py / corpus.py directly under this folder)
└── safetensors/
    ├── fnn_gossip_desc/CROWN-CROWN/#6/CROWN-CROWN        # Trained checkpoint
    └── vocabulary-3-MIND-32-64-fnn_gossip_desc.json      # Word vocabulary
```

---

## Running the Pipeline

Run all commands from the project root. On Windows PowerShell, prefix each command with `$env:KMP_DUPLICATE_LIB_OK='TRUE'`.

### 1 — Prepare data
```bash
python -X utf8 crystal/prepare_data.py          # Source CSV → articles.jsonl / users.jsonl
```

### 2 — Veracity prediction (RoBERTa)
```bash
python -X utf8 roberta_veracity/train_kfold.py  # 10-fold training + temperature scaling
python -X utf8 crystal/veracity/predict.py      # Attach pred_prob_fake / final_label to all articles
```

### 3 — Recommendation feed generation (CROWN)
```bash
python -X utf8 crystal/recsys/infer.py --k 50   # Pre-compute feeds.jsonl
```

### 4 — Deceptive keyword & sentence highlighting
```bash
python -X utf8 crystal/keyword/mask_names.py                                   # Person-name masking
python -X utf8 roberta_veracity/train_masked.py                               # Attribution model
python -X utf8 crystal/keyword/contextcite.py --model roberta_veracity/folds/fold0_masked/best \
       --ids ids.json --masked crystal/data/keyword/masked.csv                # Sentence attribution
python -X utf8 crystal/keyword/build_highlights.py                            # cc_keywords.jsonl
```
> The earlier IG-based highlighting is `python -X utf8 crystal/keyword/run.py`. See `crystal/keyword/README.md` for details.

### 5 — Event matching
```bash
python -X utf8 crystal/event/match.py --method cluster --threshold 0.7 --apply
```

### 6 — Pre-generate rationales (optional)
```bash
set GEMINI_API_KEY=...   # or .env
python -X utf8 -m crystal.rationale.generate --users 721944259,586105624 --pages 2
```

---

## Running the Web Demo

```bash
set GEMINI_API_KEY=...        # or .env (without it, only cached rationales are shown)
set CRYSTAL_KEYWORDS=cc       # ContextCite + cue-lexicon highlighting (default ig = earlier IG highlighting)
python -X utf8 -m uvicorn crystal.web.app:app --port 8765
```

If `crystal/data/` is present it serves the real data; otherwise it falls back to the sample data in `crystal/web/sample/`. → http://127.0.0.1:8765/

### Docker

Demo image bundling the data and cache: [oensang166/crystal](https://hub.docker.com/r/oensang166/crystal)

```bash
docker pull oensang166/crystal:latest
docker run -e GEMINI_API_KEY=... -p 8765:8765 oensang166/crystal:latest
```

Build it yourself: `docker build -t crystal .` (the Dockerfile sets `CRYSTAL_KEYWORDS=cc` and the flash-lite model as defaults).

---

## Citation

Seongeun Ryu, Hwijin Bae, Yusuk Lee, and Sang-Wook Kim. 2027.
CRYSTAL: A Credibility-aware News Recommendation System for Explainable Fake News Mitigation.
In *Companion Proceedings of the ACM Web Conference 2027 (WWW Companion '27)*,
May 10–14, 2027, Dublin, Ireland. ACM, New York, NY, USA, 4 pages.

```bibtex
@inproceedings{ryu2027crystal,
  title     = {CRYSTAL: A Credibility-aware News Recommendation System for Explainable Fake News Mitigation},
  author    = {Ryu, Seongeun and Bae, Hwijin and Lee, Yusuk and Kim, Sang-Wook},
  booktitle = {Companion Proceedings of the ACM Web Conference 2027 (WWW Companion '27)},
  year      = {2027},
  address   = {Dublin, Ireland},
  note      = {Demo paper}
}
```
