# 열처리 공정 온도 모니터링 · 조기경보 — 2025 제5회 K-인공지능 제조데이터 분석 경진대회

> **프로젝트명** 불량률 감소를 위한 실시간 공정 온도 모니터링 및 온디바이스 AI 모델 기반 조기경보 시스템
> **대회** 2025 제5회 K-인공지능 제조데이터 분석 경진대회 (KAMP, 한국과학기술원 주관) · **팀** Bigbase (3인)
> **데이터** KAMP 「열처리 공정최적화 AI 데이터셋」 — 136개 생산 배치(배정번호), 약 293만 행의 1초 단위 공정 센서 데이터 + 품질 검사 결과

열처리 공정은 배치마다 기계 온도 세팅이 달라 불량률을 예측하기 어려움. 이 프로젝트는 **공개 제조 데이터**로
(1) 불량률이 0% 인 배치와 상위 10개 배치의 공정 온도 차이를 통계적으로 검정해 핵심 공정 변수 3개와 정상 운영 범위를 도출하고,
(2) 그 변수들의 10분 후 온도를 예측하는 시계열 모델(DLinear · NLinear · LSTM)을 비교 학습한 뒤,
(3) 최적 모델을 ONNX 로 변환해 Flask 실시간 대시보드에서 슬라이딩 윈도우 추론과 임계값 알람을 수행함.

## 역할 분담

| 단계 | 내용 | 산출물 | 담당 |
|---|---|---|---|
| 데이터 선정 · 탐색 | KAMP 데이터셋 3종(소성가공 · 사출성형 · 열처리) EDA, 사출성형 수주량 예측 베이스라인 | `exploration/` | 최지혁 |
| 1. 전처리 · 통계 분석 | 결측 제거 · 선형 보간 · 10초 리샘플링, 배정번호별 최고 · 최저 온도 추출, Welch t-test · Cohen's d · Mann-Whitney U 로 핵심 변수 3개 선정 | `analyze.ipynb`, `data/unified.csv`, `data/analyze.csv` | 최지혁 |
| 2. 시계열 예측 모델 | 배정번호 단위 슬라이딩 윈도우 데이터셋, DLinear · NLinear · LSTM × 학습률 2 × 손실함수 2 = 36개 조합 실험, MAE · RMSE · SMAPE 평가 | `predict.ipynb`, `models/`, `result.csv`, `scaler.pkl` | 최지혁 |
| 3. ONNX 변환 · 대시보드 | 최적 모델 ONNX export, Flask 서버(1초 스트리밍 · 1분 주기 추론 · 임계값 알람), Chart.js 대시보드 | `onnx/`, `dashboard.ipynb`, `templates/` | 팀원 |

이 저장소는 팀 제출물 전체를 담되, 3단계(ONNX · Flask)는 팀원이 구현한 코드임을 명시함.

## 저장소 구성

```
.
├── analyze.ipynb              1단계: 통계 분석 (핵심 공정 변수 · 정상 운영 범위 도출)
├── predict.ipynb              2단계: 시계열 예측 모델 학습 · 36개 조합 비교
├── dashboard.ipynb            3단계: 실시간 모니터링 대시보드 (Flask + ONNX Runtime)
├── models/                    DLinear.py · NLinear.py · LSTM.py
├── onnx/                      학습된 최적 LSTM 모델 3개 (타겟 변수별)
├── templates/                 main.html(예측 · 알람) · dashboard.html(14개 공정 변수)
├── data/                      unified.csv(배치별 불량률 · 온도 요약) · analyze.csv(검정 결과) · label.csv
│                              
├── result.csv                 36개 조합 성능 표
├── exploration/               데이터셋 선정 단계: 소성가공 · 사출성형 EDA, 사출성형 수주량 예측
│   ├── eda_dataset_selection.ipynb
│   ├── injection_order_forecasting.ipynb   (원-핫 vs 제품 임베딩 신경망, 차분 예측, XGBoost, HDBSCAN 군집별 모델)
│   ├── model.py                            (제품 임베딩 신경망 정의)
│   └── data/sample.csv · product_order_stats.csv
└── requirements.txt
```

## 데이터 준비

1. KAMP(인공지능 중소벤처 제조 플랫폼, https://www.kamp-ai.kr) 에서 **열처리 공정최적화 AI 데이터셋**을 내려받는다.
2. `품질전처리후데이터.csv`(센서, 1초 단위 · 16개 컬럼) 와 `label.xlsx`(배정번호별 양품 · 불량 수량) 를 `data/` 에 둔다.
3. `exploration/` 노트북은 사출성형 공급망최적화 · 소성가공 품질보증 데이터셋을 `exploration/data/` 에 둔 뒤 실행한다 (`sample.csv` 는 사출성형 데이터 100행 예시).

원본 데이터는 KAMP 이용약관에 따라 저장소에 포함하지 않음.

## 실행

```bash
pip install -r requirements.txt            # pandas · scipy · scikit-learn · torch · onnxruntime · flask · matplotlib

jupyter notebook analyze.ipynb             # 1단계 → data/unified.csv, data/analyze.csv
jupyter notebook predict.ipynb             # 2단계 → onnx/*.onnx, result.csv, scaler.pkl
jupyter notebook dashboard.ipynb           # 3단계 → http://127.0.0.1:5432
```

## 방법 요약

**전처리.** 모든 공정 변수가 결측인 4,677행 제거(완전성 99.84% → 100%) 후 배정번호별로 1초 연속 시간축을 만들어 선형 보간, 10초 평균으로 리샘플링(크기 90% 감소). 센서 데이터와 품질 데이터를 배정번호로 병합해 불량률(불량수량 / 총수량)을 계산.

**핵심 공정 변수 식별.** 불량률 0% 그룹(3개 배치) 과 불량률 상위 10개 그룹의 변수별 최고 · 최저 온도를 비교. t-test p-value < 0.05 이면서 |Cohen's d| 가 큰 순으로 3개 선정.

| 순위 | 변수 | Cohen's d | p-value | 정상 운영 범위 |
|---|---|---|---|---|
| 1 | 소입로 온도 1 Zone (최저) | −2.31 | 0.014 | 850.13 °C 이상 |
| 2 | 건조로 온도 2 Zone (최저) | −2.15 | 0.007 | 98.88 °C 이상 |
| 3 | 소입로 온도 4 Zone (최저) | +1.68 | 0.028 | 858.71 °C 이하 |

**시계열 예측.** 입력 30스텝(10초 × 30 = 5분) → 출력 60스텝(10분) 단변량 예측. 배정번호 기준 시간순 70 / 10 / 20 분할, 훈련 통계로 표준화, 배정번호 경계를 넘지 않는 슬라이딩 윈도우. Adam, epoch 10, batch 4,096, 학습률 {0.001, 0.01} × 손실 {L1, MSE}.

| 타겟 변수 | 최적 모델 | MAE | RMSE |
|---|---|---|---|
| 건조로 온도 2 Zone | LSTM (lr 0.01, L1) | 0.334 | 0.563 |
| 소입로 온도 1 Zone | LSTM (lr 0.01, L1) | 0.325 | 0.669 |
| 소입로 온도 4 Zone | LSTM (lr 0.01, L1) | 0.312 | 0.456 |

**실시간 모니터링(팀원 구현).** 마지막 배정번호 데이터를 1초 간격으로 재생, 처음 300초는 warm-up, 이후 60초마다 최근 300초를 10초로 리샘플링해 ONNX 모델로 60스텝 예측. 예측값 중 하나라도 정상 범위를 벗어나면 변수별 알람 램프 점등.
