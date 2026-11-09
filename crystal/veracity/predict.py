"""K-fold 교차 예측(roberta_veracity/kfold_predictions.csv, train_kfold.py)을 읽어 전체 기사에 pred_label, pred_prob_fake(보정), factcheck(null),
veracity_source("model"), final_label, fold를 붙이고 split은 전건 "test"(모든 기사가 자기를 학습에 안 쓴 fold 모델의 판정)로 두어 articles.jsonl을 같은 순서로 덮어쓴다.
실행: python -X utf8 crystal/veracity/predict.py"""
import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH = os.path.join(ROOT, "crystal", "data", "articles.jsonl")
PRED = os.path.join(ROOT, "roberta_veracity", "kfold_predictions.csv")


def main():
    with open(PRED, encoding="utf-8", newline="") as f:
        pred = {r["news_id"]: r for r in csv.DictReader(f)}
    rows = [json.loads(l) for l in open(PATH, encoding="utf-8")]
    for r in rows:
        p = pred[r["article_id"]]
        r.update({"split": "test", "fold": int(p["fold"]), "pred_label": int(p["pred"]), "pred_prob_fake": round(float(p["prob_fake"]), 6),
                  "factcheck": None, "veracity_source": "model", "final_label": int(p["pred"])})
    tmp = PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, PATH)
    acc = sum(r["pred_label"] == r["label"] for r in rows) / len(rows)
    print(f"done {len(rows)} rows, pooled acc {acc:.4f}")


if __name__ == "__main__":
    main()
