# -*- coding: utf-8 -*-
"""
07_compare_results.py
full_ft_results.json 과 peft_lora_results.json 을 불러와서
학습 시간 / 모델 용량 / 정확도(accuracy, macro F1) / 학습 가능 파라미터 비율을
표와 그래프로 비교합니다.

사용법:
    05_train_full_finetune.py, 06_train_peft_lora.py 를 각각 실행해서
    두 결과 json이 모두 있는 폴더에서 실행하세요.
    pip install matplotlib pandas
"""

import json
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"  # macOS면 "AppleGothic", Linux면 "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

FULL_FT_JSON = "full_ft_results.json"
PEFT_JSON = "peft_lora_results.json"
OUT_DIR = "figures"


def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(FULL_FT_JSON, encoding="utf-8") as f:
        full_ft = json.load(f)
    with open(PEFT_JSON, encoding="utf-8") as f:
        peft = json.load(f)

    # ── 1) 비교표 ────────────────────────────────────────────────
    rows = [
        {"지표": "Test Accuracy", "Full FT": full_ft["test_accuracy"], "PEFT(LoRA)": peft["test_accuracy"]},
        {"지표": "Test F1(macro)", "Full FT": full_ft["test_f1_macro"], "PEFT(LoRA)": peft["test_f1_macro"]},
        {"지표": "학습 시간(초)", "Full FT": full_ft["train_time_sec"], "PEFT(LoRA)": peft["train_time_sec"]},
        {"지표": "모델 용량(MB)", "Full FT": full_ft["model_size_mb"], "PEFT(LoRA)": peft["model_size_mb"]},
        {"지표": "학습 가능 파라미터 수", "Full FT": full_ft["trainable_params"], "PEFT(LoRA)": peft["trainable_params"]},
        {"지표": "학습 가능 파라미터 비율(%)", "Full FT": full_ft["trainable_ratio_pct"], "PEFT(LoRA)": peft["trainable_ratio_pct"]},
    ]
    df = pd.DataFrame(rows)
    df["차이 (PEFT - FullFT)"] = df["PEFT(LoRA)"] - df["Full FT"]
    df["비율 (PEFT / FullFT)"] = (df["PEFT(LoRA)"] / df["Full FT"]).round(4)

    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    print("=== Full FT vs PEFT(LoRA) 비교표 ===")
    print(df.to_string(index=False))
    df.to_csv("comparison_table.csv", index=False, encoding="utf-8-sig")

    # ── 2) 학습 시간 비교 ────────────────────────────────────────
    plt.figure(figsize=(5, 5))
    methods = ["Full FT", "PEFT(LoRA)"]
    times = [full_ft["train_time_sec"] / 60, peft["train_time_sec"] / 60]
    bars = plt.bar(methods, times, color=["#4C72B0", "#DD8452"])
    plt.ylabel("학습 시간 (분)")
    plt.title("학습 시간 비교")
    for b, t in zip(bars, times):
        plt.text(b.get_x() + b.get_width() / 2, t, f"{t:.1f}분", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/compare_train_time.png", dpi=150)
    plt.close()

    # ── 3) 모델 용량 비교 (로그 스케일: 차이가 커서 선형으로는 잘 안 보임) ─
    plt.figure(figsize=(5, 5))
    sizes = [full_ft["model_size_mb"], peft["model_size_mb"]]
    bars = plt.bar(methods, sizes, color=["#4C72B0", "#DD8452"])
    plt.yscale("log")
    plt.ylabel("모델 용량 (MB, 로그 스케일)")
    plt.title("저장 모델 용량 비교")
    for b, s in zip(bars, sizes):
        plt.text(b.get_x() + b.get_width() / 2, s, f"{s:.1f}MB", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/compare_model_size.png", dpi=150)
    plt.close()

    # ── 4) 정확도 / F1 비교 ─────────────────────────────────────
    plt.figure(figsize=(6, 5))
    x = range(len(methods))
    width = 0.35
    accs = [full_ft["test_accuracy"], peft["test_accuracy"]]
    f1s = [full_ft["test_f1_macro"], peft["test_f1_macro"]]
    plt.bar([i - width / 2 for i in x], accs, width, label="Accuracy", color="#4C72B0")
    plt.bar([i + width / 2 for i in x], f1s, width, label="F1(macro)", color="#DD8452")
    plt.xticks(list(x), methods)
    plt.ylim(0, 1)
    plt.ylabel("점수")
    plt.title("성능 비교 (Accuracy vs F1-macro)")
    plt.legend()
    for i, (a, f1) in enumerate(zip(accs, f1s)):
        plt.text(i - width / 2, a, f"{a:.3f}", ha="center", va="bottom", fontsize=9)
        plt.text(i + width / 2, f1, f"{f1:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/compare_performance.png", dpi=150)
    plt.close()

    # ── 5) 학습 가능 파라미터 비율 비교 (로그 스케일) ──────────────
    plt.figure(figsize=(5, 5))
    ratios = [full_ft["trainable_ratio_pct"], peft["trainable_ratio_pct"]]
    bars = plt.bar(methods, ratios, color=["#4C72B0", "#DD8452"])
    plt.yscale("log")
    plt.ylabel("학습 가능 파라미터 비율 (%, 로그 스케일)")
    plt.title("학습 가능 파라미터 비율 비교")
    for b, r in zip(bars, ratios):
        plt.text(b.get_x() + b.get_width() / 2, r, f"{r:.3f}%", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/compare_trainable_params.png", dpi=150)
    plt.close()

    print(f"\n그래프 4종을 './{OUT_DIR}/' 에 저장했습니다:")
    print("  - compare_train_time.png")
    print("  - compare_model_size.png")
    print("  - compare_performance.png")
    print("  - compare_trainable_params.png")
    print("comparison_table.csv 도 저장했습니다.")


if __name__ == "__main__":
    main()
