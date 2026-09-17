# 쇼핑몰/SNS 리뷰 감성분석 — Full Fine-Tuning vs PEFT(LoRA)

쇼핑몰·SNS에서 수집된 한국어 리뷰 데이터(패션/화장품/가전/IT기기/생활 5개 도메인)를 대상으로,
BERT 계열 사전학습 모델을 **전체 파라미터 파인튜닝(Full FT)** 방식과 **PEFT(LoRA)** 방식으로
각각 학습시켜 정확도·학습 시간·모델 저장 용량을 동일 조건에서 정량 비교한 프로젝트.

단순히 "LoRA가 가볍다"는 결론에 멈추지 않고, **백본 모델 선택(klue/bert-base vs
beomi/kcbert-base)**과 **도메인 구성(전체 vs 특정 도메인 제외)**을 바꿔가며 같은 비교를
반복해, 결론이 특정 설정에만 우연히 맞아떨어진 것은 아닌지 교차 검증했다.

## 파이프라인 구성

| 단계 | 파일 | 내용 |
|---|---|---|
| 1. 데이터 통합 + EDA | `eda.py` | 쇼핑몰/SNS 하위 폴더의 JSON 리뷰 파일을 재귀 탐색해 통합, 결측/중복/라벨분포 점검 |
| 2. 전처리 + 분할 | `preprocess.py` | 결측 라벨 제거, 라벨 리매핑(-1/0/1 → 0/1/2), 라벨×도메인 층화 분할(8:1:1) |
| 3. 텍스트 길이 분석 | `text_length_by_group.py` | 도메인/카테고리/Source별 문자 길이 분포 시각화 |
| 4. 토큰 길이 분석 | `token_length_check.py` | 토크나이저 기준 토큰 길이 분포 및 max_length 후보별 truncation 비율 계산 |
| 5. Full FT 학습 | `train_full_finetune.py` | 전체 파라미터 파인튜닝, 학습시간/모델용량/성능 기록 |
| 6. PEFT(LoRA) 학습 | `train_peft_lora.py` | LoRA 어댑터(r=8) 학습, Full FT와 동일 조건에서 비교 |
| 7. 결과 비교 | `compare_results.py` | 두 방식의 결과 JSON을 표/그래프로 비교 |

## 핵심 기술적 의사결정

### 1) 도메인 필터링 근거 — "생활 도메인 제외"는 감으로 정한 게 아니다
토큰 길이 분석(`token_length_check.py`) 결과, 여러 `max_length` 후보 중 **생활 도메인을
제외하면 max_length=192만으로도 truncation 비율이 1.71% 수준**으로 충분히 낮아진다는 것을
확인했다. 생활 도메인을 포함하면 같은 truncation 수준을 맞추기 위해 max_length를 256까지
올려야 해 연산 비용이 늘어난다. 학습·평가 스크립트의 `DOMAIN_FILTER`에 이 근거를 주석으로
남기고, 실제로 두 조건(전체 도메인 vs 생활 제외) 모두 실험해 비교했다.

### 2) 라벨 불균형 대응 — 커스텀 WeightedTrainer
긍정 리뷰 쏠림 현상이 있어, HuggingFace `Trainer`를 상속한 `WeightedTrainer`를 만들어
`CrossEntropyLoss`에 클래스별 가중치(빈도의 역수)를 직접 적용했다. Full FT와 PEFT 스크립트
양쪽에 동일하게 적용해 비교 조건을 통제했다.

### 3) 공정한 비교를 위한 조건 통제
Full FT와 PEFT 두 스크립트는 데이터/전처리/에폭 수(5)/배치 크기(16)/시드(42)/클래스 가중치
로직까지 전부 동일하게 맞추고, **학습 방식 자체(전체 파라미터 vs LoRA 어댑터)만 차이를 두어**
성능 차이가 다른 변수에서 비롯되지 않도록 설계했다. 학습률만 PEFT 관례치(2e-4, Full FT는 2e-5)로
다르게 뒀다.

### 4) 백본 모델 비교 — klue/bert-base vs beomi/kcbert-base
같은 비교를 backbone을 바꿔가며 반복 검증했다(전체 도메인, max_length=256, n=184,525건 기준):

| Backbone | 방식 | Test Acc | Test F1(macro) | 학습 시간 |
|---|---|---|---|---|
| klue/bert-base | Full FT | 0.9127 | 0.8825 | 3,879.7초 |
| klue/bert-base | PEFT(LoRA) | 0.9036 | 0.8778 | 2,575.1초 (33.6%↓) |
| beomi/kcbert-base | Full FT | 0.9053 | 0.8746 | 5,369.5초 |
| beomi/kcbert-base | PEFT(LoRA) | 0.8939 | 0.8664 | 3,263.2초 (39.2%↓) |

두 지표 모두 klue/bert-base가 우세했고 학습 속도도 더 빨라, 이후 실험은 klue/bert-base를
기본 backbone으로 채택했다.

### 5) 도메인 필터링 효과 검증 (생활 도메인 제외, klue/bert-base, n=179,591건)

| 방식 | Test Acc | Test F1(macro) | 학습 시간 |
|---|---|---|---|
| Full FT | 0.9087 | 0.8826 | 5,191.5초 |
| PEFT(LoRA) | 0.9063 | 0.8804 | 3,311.4초 (36.2%↓) |

max_length를 256→192로 줄였음에도(1번 항목의 truncation 분석 근거) 성능은 전체 도메인
포함 실험과 동등하거나 오히려 소폭 개선됐다. 다만 이 실행에서는 학습 시간 자체가 max_length가
더 큰 전체 도메인 실험보다 길게 측정됐는데, 이는 실행 시점의 GPU 자원 경합 등 외부 요인으로
추정되며 truncation 감소가 곧 학습 시간 단축을 보장하지는 않는다는 점을 함께 기록해 둔다.

### 6) 단일 도메인(화장품) 심층 비교 — 모델 용량/파라미터 비교의 기준 실험

| Backbone | 방식 | Test Acc | Test F1(macro) | 학습 시간 | 모델 용량 | 학습가능 파라미터 비율 |
|---|---|---|---|---|---|---|
| klue/bert-base | Full FT | 0.9287 | 0.8372 | 1,077.4초 | 422.7MB | 100% |
| klue/bert-base | PEFT(LoRA) | 0.9184 | 0.8367 | 869.4초 (19.3%↓) | 1.9MB (**222배↓**) | 0.268% |
| beomi/kcbert-base | Full FT | 0.9208 | 0.8248 | 907.2초 | 416.2MB | 100% |
| beomi/kcbert-base | PEFT(LoRA) | 0.9006 | 0.8110 | 601.2초 (33.7%↓) | 1.8MB (**231배↓**) | 0.272% |

화장품 단일 도메인 기준에서도 klue/bert-base의 Full FT-PEFT 성능 격차(F1 −0.0005)가
kcbert-base의 격차(F1 −0.0138)보다 훨씬 작아, 백본 선택이 "PEFT가 Full FT를 얼마나 잘
따라가는지"에도 영향을 준다는 것을 확인했다.

## 결론

- 4가지 backbone×도메인 조합 모두에서 PEFT(LoRA)는 **학습 가능 파라미터를 0.27% 수준으로,
  모델 저장 용량을 220배 이상 줄이면서도** Test Accuracy/F1 손실을 1%p 내외로 억제했다.
- 같은 비교를 backbone과 도메인 구성을 바꿔 4번 반복함으로써, 이 결론이 특정 설정에만
  우연히 맞은 결과가 아님을 확인했다.
- 리소스 제약이 있는 환경(저장 공간, 여러 도메인/버전 모델을 동시에 서빙해야 하는 경우 등)에서
  PEFT/LoRA가 실용적인 대안이 될 수 있음을 실측으로 뒷받침했다.

## 실행 방법

```bash
pip install pandas scikit-learn matplotlib transformers datasets torch accelerate peft

python eda.py                    # 1단계: JSON 리뷰 통합 + EDA (combined_reviews.csv 생성)
python preprocess.py             # 2단계: 전처리 + train/valid/test 분할
python text_length_by_group.py   # 3단계: 텍스트 길이 시각화 (figures/ 에 저장)
python token_length_check.py     # 4단계: 토큰 길이 분석 + max_length 결정
python train_full_finetune.py    # 5단계: Full Fine-Tuning 학습
python train_peft_lora.py        # 6단계: PEFT(LoRA) 학습
python compare_results.py        # 7단계: 결과 비교 표/그래프 생성
```

원본 리뷰 데이터(JSON)와 통합 CSV, 학습된 모델/체크포인트는 용량 문제로 저장소에 포함하지
않았다. `eda.py` 상단의 `ROOT_DIR`을 리뷰 데이터가 있는 경로로 지정하면 파이프라인을
그대로 재현할 수 있다.

## 기술 스택

klue/bert-base · beomi/kcbert-base · HuggingFace Transformers/Datasets · PEFT(LoRA) ·
scikit-learn · pandas · matplotlib
