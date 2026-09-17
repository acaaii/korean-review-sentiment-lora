# -*- coding: utf-8 -*-
"""
05_train_full_finetune.py
Full Fine-Tuning 방식으로 감성 분석 모델을 학습합니다.
- train.csv / valid.csv / test.csv 를 불러와 DOMAIN_FILTER로 필터링
- klue/bert-base 전체 파라미터를 학습
- 학습 시간, 저장 모델 용량, test set 성능(accuracy/macro F1)을 기록해서
  이후 PEFT 스크립트와 비교할 수 있도록 results json으로 저장

사용법:
    02_preprocess.py 로 만든 train.csv/valid.csv/test.csv/label_map.json 이 있는 폴더에서 실행하세요.
    pip install transformers datasets scikit-learn torch accelerate
"""

import os
import json
import time
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)

# ── 설정 ──────────────────────────────────────────────────────────────
MODEL_NAME = "klue/bert-base"
DOMAIN_FILTER = ["패션", "화장품", "가전", "IT기기"]   # 생활 제외
MAX_LENGTH = 192                    # 생활 제외 시 truncation 1.71% 수준 (klue 실측은 아니나 안전한 값)
BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5
USE_CLASS_WEIGHTS = True          # 라벨 불균형(긍정 쏠림) 대응
SEED = 42

RUN_TAG = "klue_nolife"                     # 모델x도메인 조합별로 결과 구분
OUTPUT_DIR = f"full_ft_model_{RUN_TAG}"          # 최종 모델 저장 위치
CHECKPOINT_DIR = f"full_ft_checkpoints_{RUN_TAG}"  # 학습 중 체크포인트 위치 (용량 비교와는 별개)
RESULTS_JSON = f"full_ft_results_{RUN_TAG}.json"

set_seed(SEED)


def load_split(name: str) -> pd.DataFrame:
    df = pd.read_csv(f"{name}.csv", encoding="utf-8-sig")
    if DOMAIN_FILTER is not None:
        if isinstance(DOMAIN_FILTER, (list, tuple, set)):
            df = df[df["Domain"].isin(DOMAIN_FILTER)].copy()
        else:
            df = df[df["Domain"] == DOMAIN_FILTER].copy()
    return df.reset_index(drop=True)


def get_dir_size_mb(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / (1024 ** 2)


class WeightedTrainer(Trainer):
    """라벨 불균형 대응을 위해 CrossEntropyLoss에 class weight를 적용하는 Trainer."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def main():
    # ── 1) 데이터 로드 ────────────────────────────────────────────
    train_df = load_split("train")
    valid_df = load_split("valid")
    test_df = load_split("test")
    print(f"train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)} "
          f"(DOMAIN_FILTER={DOMAIN_FILTER})")

    with open("label_map.json", encoding="utf-8") as f:
        label_map = json.load(f)
    id2label = {int(k): v for k, v in label_map["id2label"].items()}
    label2id = {v: int(k) for k, v in id2label.items()}
    num_labels = len(id2label)
    print("라벨 매핑:", id2label)

    # ── 2) 토크나이저 / 데이터셋 변환 ────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]])
    valid_ds = Dataset.from_pandas(valid_df[["text", "label"]])
    test_ds = Dataset.from_pandas(test_df[["text", "label"]])

    train_ds = train_ds.map(tokenize_fn, batched=True)
    valid_ds = valid_ds.map(tokenize_fn, batched=True)
    test_ds = test_ds.map(tokenize_fn, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ── 3) 클래스 가중치 계산 (train 기준) ──────────────────────
    class_weights = None
    if USE_CLASS_WEIGHTS:
        counts = train_df["label"].value_counts().sort_index()
        counts = counts.reindex(range(num_labels), fill_value=0)
        weights = (counts.sum() / (num_labels * counts)).values
        class_weights = torch.tensor(weights, dtype=torch.float)
        print("클래스 가중치:", dict(zip(range(num_labels), weights.round(3))))

    # ── 4) 모델 로드 (Full Fine-Tuning: 전체 파라미터 학습 가능 상태 그대로) ─
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels, id2label=id2label, label2id=label2id,
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"전체 파라미터: {total_params:,} / 학습 가능 파라미터: {trainable_params:,} "
          f"({trainable_params / total_params * 100:.1f}%)")

    # ── 5) 학습 설정 ─────────────────────────────────────────────
    args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),   # GPU 있으면 mixed precision으로 속도 향상
        report_to="none",
        seed=SEED,
    )

    # transformers 버전에 따라 Trainer의 토크나이저 전달 인자명이 다릅니다.
    # (구버전: tokenizer=, 신버전(4.46+): processing_class=)
    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )
    try:
        trainer = WeightedTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = WeightedTrainer(tokenizer=tokenizer, **trainer_kwargs)

    # ── 6) 학습 (시간 측정) ──────────────────────────────────────
    print("\n=== Full Fine-Tuning 학습 시작 ===")
    start_time = time.time()
    trainer.train()
    train_time_sec = time.time() - start_time
    print(f"\n학습 소요 시간: {train_time_sec:.1f}초 ({train_time_sec / 60:.1f}분)")

    # ── 7) 최종 모델 저장 (체크포인트 말고 최종본만 별도 디렉토리에) ─
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    model_size_mb = get_dir_size_mb(OUTPUT_DIR)
    print(f"저장된 모델 용량: {model_size_mb:.1f} MB ({OUTPUT_DIR})")

    # ── 8) Test set 평가 ─────────────────────────────────────────
    print("\n=== Test set 평가 ===")
    pred_output = trainer.predict(test_ds)
    preds = np.argmax(pred_output.predictions, axis=-1)
    labels = pred_output.label_ids

    test_acc = accuracy_score(labels, preds)
    test_f1_macro = f1_score(labels, preds, average="macro")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1(macro): {test_f1_macro:.4f}")
    print("\n분류 리포트:")
    target_names = [id2label[i] for i in range(num_labels)]
    report = classification_report(labels, preds, target_names=target_names, digits=4)
    print(report)

    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix (행=실제, 열=예측):")
    print(pd.DataFrame(cm, index=target_names, columns=target_names))

    # ── 9) 결과 저장 (PEFT 스크립트와 비교용) ───────────────────
    results = {
        "method": "full_finetuning",
        "model_name": MODEL_NAME,
        "domain_filter": DOMAIN_FILTER,
        "max_length": MAX_LENGTH,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio_pct": round(trainable_params / total_params * 100, 4),
        "train_time_sec": round(train_time_sec, 1),
        "model_size_mb": round(model_size_mb, 1),
        "test_accuracy": round(test_acc, 4),
        "test_f1_macro": round(test_f1_macro, 4),
        "n_train": len(train_df),
        "n_valid": len(valid_df),
        "n_test": len(test_df),
    }
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n결과를 '{RESULTS_JSON}' 에 저장했습니다.")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
