# -*- coding: utf-8 -*-
"""
06_train_peft_lora.py
PEFT(LoRA) 방식으로 감성 분석 모델을 학습합니다.
05_train_full_finetune.py 와 데이터/전처리/epoch/batch size/class weight 등
모든 조건을 동일하게 맞추고, 학습 방식(전체 파라미터 vs LoRA 어댑터)만 다르게 해서
공정한 비교가 되도록 구성했습니다.

사용법:
    05_train_full_finetune.py 와 동일한 폴더/데이터에서 실행하세요.
    pip install transformers datasets scikit-learn torch accelerate peft
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
from peft import LoraConfig, get_peft_model, TaskType

# ── 설정 (05_train_full_finetune.py와 동일하게 유지) ───────────────────
MODEL_NAME = "klue/bert-base"
DOMAIN_FILTER = ["패션", "화장품", "가전", "IT기기"]   # 생활 제외
MAX_LENGTH = 192
BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 2e-4        # PEFT는 학습 가능 파라미터가 적어 보통 Full FT보다 높은 LR을 씀 (관례치)
USE_CLASS_WEIGHTS = True
SEED = 42

# ── LoRA 설정 ────────────────────────────────────────────────────────
LORA_R = 8                  # rank: 어댑터 행렬의 차원. 클수록 표현력↑, 파라미터/용량↑
LORA_ALPHA = 16              # 스케일링 계수, 보통 r의 2배 정도로 설정
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["query", "value"]  # BERT의 self-attention 내 Linear 모듈 이름

RUN_TAG = "klue_nolife"
OUTPUT_DIR = f"peft_lora_model_{RUN_TAG}"
CHECKPOINT_DIR = f"peft_lora_checkpoints_{RUN_TAG}"
RESULTS_JSON = f"peft_lora_results_{RUN_TAG}.json"

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
    # ── 1) 데이터 로드 (Full FT와 완전히 동일) ──────────────────
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

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]]).map(tokenize_fn, batched=True)
    valid_ds = Dataset.from_pandas(valid_df[["text", "label"]]).map(tokenize_fn, batched=True)
    test_ds = Dataset.from_pandas(test_df[["text", "label"]]).map(tokenize_fn, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    class_weights = None
    if USE_CLASS_WEIGHTS:
        counts = train_df["label"].value_counts().sort_index()
        counts = counts.reindex(range(num_labels), fill_value=0)
        weights = (counts.sum() / (num_labels * counts)).values
        class_weights = torch.tensor(weights, dtype=torch.float)
        print("클래스 가중치:", dict(zip(range(num_labels), weights.round(3))))

    # ── 2) 베이스 모델 로드 후 LoRA 어댑터 부착 ──────────────────
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels, id2label=id2label, label2id=label2id,
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"전체 파라미터: {total_params:,} / 학습 가능 파라미터: {trainable_params:,} "
          f"({trainable_params / total_params * 100:.4f}%)")
    model.print_trainable_parameters()

    # ── 3) 학습 설정 (Full FT와 동일한 epoch/batch size, LR만 PEFT 관례치) ─
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
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=SEED,
        label_names=["labels"],   # PEFT 모델은 자동 추론이 안 될 때가 있어 명시
    )

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

    # ── 4) 학습 (시간 측정) ──────────────────────────────────────
    print("\n=== PEFT(LoRA) 학습 시작 ===")
    start_time = time.time()
    trainer.train()
    train_time_sec = time.time() - start_time
    print(f"\n학습 소요 시간: {train_time_sec:.1f}초 ({train_time_sec / 60:.1f}분)")

    # ── 5) 최종 모델(어댑터) 저장 ────────────────────────────────
    # PEFT는 save_pretrained 시 어댑터 가중치만 저장됨 (베이스 모델은 저장 안 함)
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    model_size_mb = get_dir_size_mb(OUTPUT_DIR)
    print(f"저장된 어댑터(+토크나이저) 용량: {model_size_mb:.1f} MB ({OUTPUT_DIR})")

    # ── 6) Test set 평가 ─────────────────────────────────────────
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

    # ── 7) 결과 저장 ─────────────────────────────────────────────
    results = {
        "method": "peft_lora",
        "model_name": MODEL_NAME,
        "domain_filter": DOMAIN_FILTER,
        "max_length": MAX_LENGTH,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_target_modules": LORA_TARGET_MODULES,
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
