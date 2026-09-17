# -*- coding: utf-8 -*-
"""
03_text_length_by_group.py
combined_reviews.csv(01_eda.py 결과물)를 읽어서
Domain별, MainCategory별 텍스트 길이 분포를 그래프로 그립니다.

- 박스플롯: 도메인별, 카테고리별 길이 분포(중앙값/사분위/이상치)를 한눈에 비교
- 히스토그램: 도메인별 길이 분포 모양(치우침 정도)을 비교
- Source(쇼핑몰 vs SNS)별 비교도 함께 확인

사용법:
    01_eda.py 를 먼저 실행해서 combined_reviews.csv 를 만들어 둔 뒤, 같은 폴더에서 실행하세요.
    pip install matplotlib seaborn pandas
    출력 이미지는 현재 폴더의 ./figures/ 아래 저장됩니다.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# 한글 폰트 설정 (Windows 기준 맑은 고딕. 다른 OS면 폰트명 수정 필요)
plt.rcParams["font.family"] = "Malgun Gothic"   # macOS면 "AppleGothic", Linux면 "NanumGothic" 등으로 변경
plt.rcParams["axes.unicode_minus"] = False       # 마이너스 기호 깨짐 방지

INPUT_CSV = "combined_reviews.csv"
OUT_DIR = "figures"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    df = df.dropna(subset=["RawText"]).copy()
    df["text_len"] = df["RawText"].astype(str).str.len()

    # 극단적 이상치 때문에 그래프가 눌리는 것을 방지하기 위해 상위 1% 컷 버전도 별도로 사용
    p99 = df["text_len"].quantile(0.99)
    print(f"길이 99 퍼센타일: {p99:.0f}자 (그래프의 y축 상한 참고용)")

    # ── 1) 도메인별 박스플롯 ─────────────────────────────────────
    domain_order = df.groupby("Domain")["text_len"].median().sort_values(ascending=False).index

    plt.figure(figsize=(9, 6))
    data_by_domain = [df.loc[df["Domain"] == d, "text_len"] for d in domain_order]
    plt.boxplot(data_by_domain, labels=domain_order, showfliers=False)
    plt.title("도메인별 리뷰 텍스트 길이 분포 (이상치 제외)")
    plt.ylabel("텍스트 길이 (문자 수)")
    plt.xlabel("Domain")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "length_by_domain_boxplot.png"), dpi=150)
    plt.close()

    # ── 2) MainCategory별 박스플롯 (카테고리 수가 많아 세로로 길게) ─
    cat_order = df.groupby("MainCategory")["text_len"].median().sort_values(ascending=False).index

    plt.figure(figsize=(9, max(6, 0.35 * len(cat_order))))
    data_by_cat = [df.loc[df["MainCategory"] == c, "text_len"] for c in cat_order]
    plt.boxplot(data_by_cat, labels=cat_order, vert=False, showfliers=False)
    plt.title("카테고리별 리뷰 텍스트 길이 분포 (이상치 제외)")
    plt.xlabel("텍스트 길이 (문자 수)")
    plt.ylabel("MainCategory")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "length_by_category_boxplot.png"), dpi=150)
    plt.close()

    # ── 3) Source(쇼핑몰 vs SNS)별 박스플롯 ─────────────────────
    plt.figure(figsize=(5, 6))
    data_by_source = [df.loc[df["Source"] == s, "text_len"] for s in df["Source"].unique()]
    plt.boxplot(data_by_source, labels=df["Source"].unique(), showfliers=False)
    plt.title("Source별 리뷰 텍스트 길이 분포 (이상치 제외)")
    plt.ylabel("텍스트 길이 (문자 수)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "length_by_source_boxplot.png"), dpi=150)
    plt.close()

    # ── 4) 도메인별 히스토그램 (분포 모양 비교, 99% 지점에서 컷) ───
    domains = df["Domain"].unique()
    fig, axes = plt.subplots(len(domains), 1, figsize=(8, 3 * len(domains)), sharex=True)
    if len(domains) == 1:
        axes = [axes]
    for ax, d in zip(axes, sorted(domains)):
        sub = df.loc[df["Domain"] == d, "text_len"]
        sub_clipped = sub[sub <= p99]
        ax.hist(sub_clipped, bins=50)
        ax.set_title(f"{d} (n={len(sub)})")
        ax.set_ylabel("빈도")
    axes[-1].set_xlabel(f"텍스트 길이 (문자 수, {p99:.0f}자 이하만 표시)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "length_hist_by_domain.png"), dpi=150)
    plt.close()

    # ── 5) 도메인별 길이 기술통계 표 출력 + CSV 저장 ───────────────
    stats = df.groupby("Domain")["text_len"].agg(
        count="count", mean="mean", median="median",
        p75=lambda x: x.quantile(0.75), p90=lambda x: x.quantile(0.90), max="max",
    )
    print("\n=== 도메인별 텍스트 길이 통계 ===")
    print(stats.round(1))
    stats.to_csv(os.path.join(OUT_DIR, "length_stats_by_domain.csv"), encoding="utf-8-sig")

    cat_stats = df.groupby("MainCategory")["text_len"].agg(
        count="count", mean="mean", median="median",
        p75=lambda x: x.quantile(0.75), p90=lambda x: x.quantile(0.90), max="max",
    ).sort_values("median", ascending=False)
    print("\n=== 카테고리별 텍스트 길이 통계 ===")
    print(cat_stats.round(1))
    cat_stats.to_csv(os.path.join(OUT_DIR, "length_stats_by_category.csv"), encoding="utf-8-sig")

    print(f"\n그래프/통계 저장 완료: ./{OUT_DIR}/ 폴더 확인하세요.")


if __name__ == "__main__":
    main()
