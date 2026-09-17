# -*- coding: utf-8 -*-
"""
02_preprocess.py
combined_reviews.csv(01_eda.py 결과물)를 읽어서
- 결측 라벨 제거
- 라벨 리매핑 (-1/0/1 -> 0/1/2)
- 텍스트 클리닝(가벼운 수준)
- train / valid / test 로 stratified split (라벨 + 도메인 기준)
- 이후 Full FT / PEFT 스크립트에서 그대로 쓸 수 있는 CSV로 저장

사용법:
    01_eda.py 를 먼저 실행해서 combined_reviews.csv 를 만들어 둔 뒤, 같은 폴더에서 실행하세요.
"""

import re
import json
import pandas as pd
from sklearn.model_selection import train_test_split

# ── 설정 ──────────────────────────────────────────────────────────────
INPUT_CSV = "combined_reviews.csv"

TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42

# 라벨 매핑: 모델 학습용 정수 라벨 (0/1/2) <-> 원래 GeneralPolarity 값
# 순서를 부정(0) < 중립(1) < 긍정(2) 로 둬서 감성의 자연스러운 순서와 일치시킴
POLARITY_TO_LABEL = {-1: 0, 0: 1, 1: 2}
LABEL_TO_NAME = {0: "부정", 1: "중립", 2: "긍정"}


def clean_text(text: str) -> str:
    """가벼운 텍스트 정제. 과도한 정제는 감성 표현(이모티콘, 반복 문자 등)을 해칠 수 있어 최소화."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    # 연속 공백/개행을 하나로
    text = re.sub(r"\s+", " ", text)
    return text


def main():
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    print(f"원본 로드: {len(df)}건")

    # ── 1) 결측 라벨 제거 ────────────────────────────────────────
    df = df.dropna(subset=["GeneralPolarity"]).copy()
    df["GeneralPolarity"] = df["GeneralPolarity"].astype(int)
    print(f"결측 라벨 제거 후: {len(df)}건")

    # ── 2) 고유 ID 재구성 (Index가 파일 간 중복되므로 file_path와 결합) ─
    df["uid"] = df["file_path"].astype(str) + "::" + df["Index"].astype(str)
    n_dup_uid = df.duplicated(subset=["uid"]).sum()
    print(f"uid(file_path+Index) 기준 중복: {n_dup_uid}건")
    df = df.drop_duplicates(subset=["uid"]).copy()

    # ── 3) 텍스트 클리닝 ──────────────────────────────────────────
    df["text"] = df["RawText"].apply(clean_text)
    before = len(df)
    df = df[df["text"].str.len() > 0].copy()
    print(f"빈 텍스트 제거: {before - len(df)}건 제거, 남은 {len(df)}건")

    # ── 4) 라벨 리매핑 ────────────────────────────────────────────
    df["label"] = df["GeneralPolarity"].map(POLARITY_TO_LABEL)
    assert df["label"].isna().sum() == 0, "매핑되지 않은 라벨이 있습니다. GeneralPolarity 값을 확인하세요."
    df["label"] = df["label"].astype(int)

    print("\n=== 최종 라벨 분포 ===")
    print(df["label"].map(LABEL_TO_NAME).value_counts())

    # ── 5) Stratify 키 생성 (라벨 x 도메인 결합) ───────────────────
    # 라벨 분포뿐 아니라 도메인 비율도 train/valid/test에서 유지되도록 함
    df["stratify_key"] = df["label"].astype(str) + "_" + df["Domain"].astype(str)

    # 너무 작은 조합(도메인 x 라벨 조합이 2건 미만)은 stratify 시 에러가 나므로 확인
    key_counts = df["stratify_key"].value_counts()
    rare_keys = key_counts[key_counts < 3].index.tolist()
    if rare_keys:
        print(f"\n[주의] 표본이 3건 미만인 (라벨,도메인) 조합 {len(rare_keys)}개는 stratify 키를 라벨만으로 완화합니다: {rare_keys}")
        df.loc[df["stratify_key"].isin(rare_keys), "stratify_key"] = df.loc[df["stratify_key"].isin(rare_keys), "label"].astype(str)

    # ── 6) train / temp 분할 후 temp를 valid / test로 분할 ─────────
    train_df, temp_df = train_test_split(
        df,
        test_size=(VALID_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED,
        stratify=df["stratify_key"],
    )

    # temp 안에서 다시 stratify (temp의 stratify_key 재사용)
    valid_size_within_temp = VALID_RATIO / (VALID_RATIO + TEST_RATIO)
    valid_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - valid_size_within_temp),
        random_state=RANDOM_SEED,
        stratify=temp_df["stratify_key"],
    )

    print(f"\n분할 결과: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")

    # ── 7) 분할별 라벨/도메인 분포 확인 (비율이 비슷한지 검증) ──────
    for name, part in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        print(f"\n--- {name} 라벨 비율(%) ---")
        print((part["label"].map(LABEL_TO_NAME).value_counts(normalize=True) * 100).round(1))

    # ── 8) 저장 (학습 스크립트에서 바로 쓸 컬럼만 정리) ─────────────
    keep_cols = ["uid", "text", "label", "GeneralPolarity", "Domain", "MainCategory", "Source"]
    train_df[keep_cols].to_csv("train.csv", index=False, encoding="utf-8-sig")
    valid_df[keep_cols].to_csv("valid.csv", index=False, encoding="utf-8-sig")
    test_df[keep_cols].to_csv("test.csv", index=False, encoding="utf-8-sig")

    with open("label_map.json", "w", encoding="utf-8") as f:
        json.dump(
            {"label2id": {v: k for k, v in LABEL_TO_NAME.items()}, "id2label": LABEL_TO_NAME},
            f, ensure_ascii=False, indent=2,
        )

    print("\ntrain.csv / valid.csv / test.csv / label_map.json 저장 완료")


if __name__ == "__main__":
    main()
