# Maple Inven 감성 분석 대시보드

## 프로젝트 개요
- **목적**: 메이플스토리 인벤 게시판(전사, 마법사, 궁수, 도적, 해적 등) 댓글 데이터를 수집하여 감성(긍정ㆍ중립ㆍ부정)을 분석하고, Streamlit 기반 웹 대시보드로 시각화합니다.  
- **주요 기능**  
  1. 게시판별ㆍ직업별 감성 비율 바 차트  
  2. 워드클라우드  
  3. 상위 15개 키워드 빈도 막대 차트  
  4. 상세 데이터 테이블

## 폴더 구조
├── app.py # Streamlit 대시보드
├── sentiment_analysis.py # 데이터 전처리ㆍ모델 학습 스크립트
├── requirements.txt # Python 패키지 목록
├── lib/
│ ├── korean_stopwords.txt # 한글 불용어
│ └── ... # 기타 헬퍼 모듈
├── data/
│ ├── posts.csv # 크롤링 원본 데이터
│ ├── posts_label.csv # 수동/자동 라벨링 데이터
│ └── posts_labeled_pred.csv # 예측 결과 포함 데이터
├── model/
│ └── 20250520_model.h5 # 학습된 Keras 모델 파일
└── README.md # (이 파일)


## 설치 및 실행

1. 가상 환경 생성 및 활성화
   bash
   # conda 환경인 경우
   conda create -n nlp-tfgpu python=3.10
   conda activate nlp-tfgpu

   # 또는 venv
   python3 -m venv .venv
   source .venv/bin/activate
2. 필수 패키지 설치
  pip install -r requirements.txt

3. 모델학습
  python sentiment_analysis.py
4. Streamlit 앱 실행
  streamlit run app.py


📝 감성 분석 파이프라인
데이터 전처리

clean_text, 토큰화, 품사 태깅, 불용어 제거, 정제

Integer Encoding, Padding

특징 추출

워드 임베딩(Tokenizer → sequences → pad_sequences)

모델 학습 (sentiment_analysis.py)

Keras Embedding + LSTM 기반 분류기

categorical_crossentropy 손실, adam 옵티마이저

학습 완료 후 model/20250520_model.h5로 저장

배치 예측 & 결과 저장

전체 데이터에 대해 model.predict

data/posts_labeled_pred.csv에 pred 컬럼 추가

📊 Streamlit 대시보드
URL: http://localhost:8501

사이드바

직업군(게시판) 필터

직업별 체크박스

메인 화면

감성 분포: 긍정/중립/부정 비율 바 차트

워드클라우드

상위 15개 키워드 빈도

python
복사
편집
from collections import Counter
tokens = " ".join(df_filtered['cleaned']).split()
top15 = Counter(tokens).most_common(15)
막대 차트로 시각화

상세 데이터 테이블: 게시판, job, 제목, pred
