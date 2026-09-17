# -*- coding: utf-8 -*-
"""
04_token_length_check.py
문자(char) 길이가 아니라, 실제 학습에 쓸 토크나이저 기준 '토큰 길이'를 확인합니다.
- 전체 / 도메인별 토큰 길이 분포
- 여러 max_length 후보(128/192/256/320/384)에 대해 "몇 %가 잘리는지(truncation 비율)"를 계산
  -> 이 표를 보고 정확도 손실과 연산비용(패딩 낭비) 사이에서 max_length를 결정하면 됩니다.

사용법:
    02_preprocess.py 로 만든 train.csv 가 있는 폴더에서 실행하세요.
    pip install transformers pandas
"""

import pandas as pd
from transformers import AutoTokenizer

# ── 설정 ──────────────────────────────────────────────────────────────
# Full FT / PEFT 실험에 쓸 모델과 동일한 것으로 맞춰야 의미가 있습니다.
MODEL_NAME = "beomi/kcbert-base"   # 나중에 "klue/bert-base", "beomi/KcELECTRA-base" 등과 비교 가능
INPUT_CSV = "train.csv"
CANDIDATE_MAX_LENGTHS = [64, 96, 128, 192, 256, 320, 384]


def main():
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"토크나이저: {MODEL_NAME}")
    print(f"대상 데이터: {len(df)}건 (train.csv)\n")

    # 특수 토큰([CLS],[SEP]) 포함해서 실제 토큰 수 계산
    token_lens = df["text"].astype(str).apply(lambda t: len(tokenizer.encode(t)))
    df["token_len"] = token_lens

    print("=== 전체 토큰 길이 기술통계 ===")
    print(df["token_len"].describe())
    print("\n백분위수 (50/75/90/95/99%):")
    print(df["token_len"].quantile([0.5, 0.75, 0.9, 0.95, 0.99]))

    print("\n=== 도메인별 토큰 길이 통계 ===")
    domain_stats = df.groupby("Domain")["token_len"].agg(
        count="count", mean="mean", median="median",
        p90=lambda x: x.quantile(0.90), p95=lambda x: x.quantile(0.95), max="max",
    )
    print(domain_stats.round(1))

    # ── max_length 후보별 truncation 비율 (전체 기준) ───────────────
    print("\n=== max_length 후보별 전체 truncation 비율 ===")
    rows = []
    for ml in CANDIDATE_MAX_LENGTHS:
        trunc_ratio = (df["token_len"] > ml).mean() * 100
        rows.append({"max_length": ml, "truncated_pct": round(trunc_ratio, 2)})
    trunc_df = pd.DataFrame(rows)
    print(trunc_df.to_string(index=False))

    # ── max_length 후보별, 도메인별 truncation 비율 (생활처럼 유독 긴 도메인이 더 잘리는지 확인) ─
    print("\n=== max_length 후보별 도메인별 truncation 비율 (%) ===")
    pivot_rows = []
    for domain, sub in df.groupby("Domain"):
        row = {"Domain": domain, "count": len(sub)}
        for ml in CANDIDATE_MAX_LENGTHS:
            row[f"trunc@{ml}"] = round((sub["token_len"] > ml).mean() * 100, 1)
        pivot_rows.append(row)
    pivot_df = pd.DataFrame(pivot_rows).sort_values("Domain")
    print(pivot_df.to_string(index=False))

    # ── 추천값 계산: 전체 truncation 비율이 5% 이하가 되는 가장 작은 후보 ─
    recommended = None
    for ml in CANDIDATE_MAX_LENGTHS:
        if (df["token_len"] > ml).mean() <= 0.05:
            recommended = ml
            break
    if recommended is None:
        recommended = CANDIDATE_MAX_LENGTHS[-1]

    print(f"\n[추천] 전체 데이터의 95% 이상을 자르지 않고 담을 수 있는 최소 max_length 후보: {recommended}")
    print("단, 위 도메인별 표에서 특정 도메인(예: 생활)만 유독 truncation이 많다면,")
    print("그 도메인 성능 저하를 감안하거나 max_length를 한 단계 더 올리는 것을 고려하세요.")

    df.to_csv("train_with_token_len.csv", index=False, encoding="utf-8-sig")
    print("\ntrain_with_token_len.csv 저장 완료 (token_len 컬럼 추가됨)")


if __name__ == "__main__":
    main()
