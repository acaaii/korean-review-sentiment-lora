# -*- coding: utf-8 -*-
"""
01_eda.py
쇼핑몰/SNS 리뷰 데이터 전체 EDA 스크립트

- ROOT_DIR 아래를 재귀적으로 탐색해서 모든 .json 파일을 찾습니다.
  (SNS/쇼핑몰 하위 폴더 depth가 서로 달라도 상관없이 동작합니다.)
- 각 리뷰를 하나의 row로 하는 DataFrame으로 합칩니다.
- 라벨 분포(전체 / Source별 / Domain별), 결측치, 텍스트 길이, 중복 등을 확인합니다.
- 결과를 combined CSV로 저장해서 이후 전처리/학습 스크립트에서 재사용합니다.

사용법:
    ROOT_DIR 값을 본인 프로젝트 경로로 맞춘 뒤 실행하세요.
    예: ROOT_DIR = r"C:/Users/uze/PycharmProjects/Sprint13"
"""

import json
import glob
import os
import pandas as pd

# ── 설정 ──────────────────────────────────────────────────────────────
ROOT_DIR = r"C:\Users\uze\PycharmProjects\Sprint13"   # <-- 본인 경로로 수정
OUTPUT_CSV = "combined_reviews.csv"

# 최종적으로 남길 표준 컬럼 (파일마다 있는 필드가 조금씩 달라서 공통 필드만 사용)
KEEP_COLS = [
    "Index", "RawText", "Source", "Domain", "MainCategory",
    "ProductName", "GeneralPolarity", "file_path", "top_folder",
]


def find_json_files(root_dir: str):
    """root_dir 아래 모든 .json 파일 경로를 재귀적으로 찾는다."""
    pattern = os.path.join(root_dir, "**", "*.json")
    files = glob.glob(pattern, recursive=True)
    return sorted(files)


def load_one_file(path: str, root_dir: str):
    """JSON 파일 하나를 읽어서 list[dict] 형태로 반환. 실패하면 빈 리스트."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] 읽기 실패: {path} ({e})")
        return []

    if isinstance(data, dict):
        # 혹시 최상위가 dict이고 내부에 리스트가 들어있는 형태일 경우 대비
        for v in data.values():
            if isinstance(v, list):
                data = v
                break
        else:
            data = [data]

    if not isinstance(data, list):
        print(f"[WARN] 예상치 못한 JSON 구조: {path} (type={type(data)})")
        return []

    # top_folder: 경로에서 SNS / 쇼핑몰 구분
    rel = os.path.relpath(path, root_dir)
    parts = rel.split(os.sep)
    top_folder = parts[0] if parts else ""

    for row in data:
        if isinstance(row, dict):
            row["file_path"] = rel
            row["top_folder"] = top_folder

    return [row for row in data if isinstance(row, dict)]


def load_all(root_dir: str) -> pd.DataFrame:
    files = find_json_files(root_dir)
    print(f"발견된 JSON 파일 수: {len(files)}")
    if len(files) == 0:
        raise FileNotFoundError(
            f"'{root_dir}' 아래에서 JSON 파일을 찾지 못했습니다. ROOT_DIR 경로를 확인하세요."
        )

    all_rows = []
    for path in files:
        all_rows.extend(load_one_file(path, root_dir))

    df = pd.DataFrame(all_rows)
    return df


def main():
    df = load_all(ROOT_DIR)
    print(f"\n총 로드된 리뷰 수(원본, 정리 전): {len(df)}")

    # 존재하지 않는 컬럼은 만들어서 NaN으로 채움 (일부 파일에 ReviewScore/RDate 없는 것과 유사한 케이스 대비)
    for col in KEEP_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[KEEP_COLS].copy()

    # ── 1) 결측치 확인 ──────────────────────────────────────────────
    print("\n=== 컬럼별 결측치 개수 ===")
    print(df.isna().sum())

    n_missing_label = df["GeneralPolarity"].isna().sum()
    n_missing_text = df["RawText"].isna().sum() | (df["RawText"].astype(str).str.strip() == "").sum()
    print(f"\nGeneralPolarity 결측: {n_missing_label}건")
    print(f"RawText 결측/빈 문자열: {n_missing_text}건")

    # ── 2) 라벨 타입 정리 (문자열 "-1","0","1" -> int) ─────────────
    def to_int_label(x):
        try:
            return int(x)
        except (TypeError, ValueError):
            return pd.NA

    df["label_int"] = df["GeneralPolarity"].apply(to_int_label)

    # 유효하지 않은 라벨(-1,0,1 이외 값)이 있는지도 체크
    valid_labels = {-1, 0, 1}
    invalid_mask = df["label_int"].notna() & (~df["label_int"].isin(valid_labels))
    print(f"\n-1/0/1 이외의 이상 라벨 값 개수: {invalid_mask.sum()}")
    if invalid_mask.sum() > 0:
        print(df.loc[invalid_mask, "GeneralPolarity"].value_counts())

    # ── 3) 라벨 분포 ─────────────────────────────────────────────
    print("\n=== 전체 라벨 분포 (label_int) ===")
    print(df["label_int"].value_counts(dropna=False).sort_index())

    print("\n=== Source(쇼핑몰/SNS)별 라벨 분포 (비율, %) ===")
    ct = pd.crosstab(df["Source"], df["label_int"], normalize="index") * 100
    print(ct.round(1))

    print("\n=== Domain별 라벨 분포 (비율, %) ===")
    ct2 = pd.crosstab(df["Domain"], df["label_int"], normalize="index") * 100
    print(ct2.round(1))

    print("\n=== top_folder(SNS/쇼핑몰 실제 폴더명) x Source 교차 확인 ===")
    print(pd.crosstab(df["top_folder"], df["Source"]))

    # ── 4) 도메인/카테고리별 규모 ───────────────────────────────
    print("\n=== Domain별 리뷰 수 ===")
    print(df["Domain"].value_counts())

    print("\n=== MainCategory별 리뷰 수 (상위 20개) ===")
    print(df["MainCategory"].value_counts().head(20))

    print("\n=== Source별 리뷰 수 ===")
    print(df["Source"].value_counts())

    # ── 5) 텍스트 길이 분포 ─────────────────────────────────────
    df["text_len"] = df["RawText"].astype(str).str.len()
    print("\n=== 텍스트 길이(문자 수) 기술통계 ===")
    print(df["text_len"].describe())
    print("\n길이 백분위수 (50/75/90/95/99%):")
    print(df["text_len"].quantile([0.5, 0.75, 0.9, 0.95, 0.99]))

    # ── 6) 중복 확인 ─────────────────────────────────────────────
    n_dup_text = df.duplicated(subset=["RawText"]).sum()
    n_dup_index = df.duplicated(subset=["Index"]).sum()
    print(f"\nRawText 기준 중복 건수: {n_dup_text}")
    print(f"Index 기준 중복 건수: {n_dup_index}")

    # ── 7) 결과 저장 ─────────────────────────────────────────────
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n전체 데이터를 '{OUTPUT_CSV}' 로 저장했습니다. (행 수: {len(df)})")


if __name__ == "__main__":
    main()
