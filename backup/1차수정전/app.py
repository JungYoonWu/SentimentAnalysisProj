import os
import re
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import pickle
from konlpy.tag import Okt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- 경로 설정 ---
BASE_DIR = os.getcwd()
# 1. 대시보드용 사전 분석 데이터
PRED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'posts_labeled_pred.csv')
# 2. 실시간 분석용 배포 모델
PROD_MODEL_DIR = os.path.join(BASE_DIR, 'production_model')
PROD_MODEL_PATH = os.path.join(PROD_MODEL_DIR, 'best_maple_model.h5')
PROD_TOKENIZER_PATH = os.path.join(PROD_MODEL_DIR, 'tokenizer.pkl')
# 3. 불용어 사전
STOPWORDS_PATH = os.path.join(BASE_DIR, 'data', 'korean_stopwords.txt')

# --- 데이터 및 모델 로딩 (캐싱) ---
@st.cache_data
def load_dashboard_data():
    """대시보드용 데이터를 로드하고 전처리합니다."""
    df = pd.read_csv(PRED_DATA_PATH, encoding='utf-8-sig')
    df['cleaned'] = (df['제목'].fillna('') + ' ' + df['본문'].fillna('')).apply(lambda x: re.sub(r"[^가-힣0-9a-zA-Z\s]", " ", str(x)))
    df['job'] = df['제목'].str.extract(r'^\[(.*?)\]').fillna('Unknown')
    return df

@st.cache_resource
def load_prediction_assets():
    """실시간 예측에 필요한 모델, 토크나이저, 불용어를 로드합니다."""
    try:
        model = load_model(PROD_MODEL_PATH)
        with open(PROD_TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        with open(STOPWORDS_PATH, 'r', encoding='utf-8') as f:
            stopwords = [line.strip() for line in f]
        okt = Okt()
        return model, tokenizer, okt, stopwords
    except FileNotFoundError:
        return None, None, None, []

# 데이터 및 모델 로드 실행
df = load_dashboard_data()
model, tokenizer, okt, stopwords = load_prediction_assets()

# --- UI 구성 ---
st.set_page_config(layout="wide")
st.title("🍁 메이플스토리 직업별 여론 대시보드 & 감성 분석")
st.write("---")

# 탭 생성
tab1, tab2 = st.tabs(["📊 대시보드", "🤖 실시간 감성 분석"])

# --- 탭 1: 대시보드 ---
with tab1:
    col1, col2 = st.columns([1, 3]) # 사이드바와 메인 영역 비율

    with col1:
        st.header("🔍 필터")
        boards = ['전체'] + sorted(df['게시판'].unique().tolist())
        sel_board = st.selectbox("게시판 선택", boards)

        if sel_board == '전체':
            jobs = sorted(df['job'].unique().tolist())
        else:
            jobs = sorted(df[df['게시판'] == sel_board]['job'].unique().tolist())

        sel_all_jobs = st.checkbox("모든 직업 선택/해제", value=True)
        if sel_all_jobs:
            sel_jobs = st.multiselect("직업 선택", jobs, default=jobs)
        else:
            sel_jobs = st.multiselect("직업 선택", jobs)

    with col2:
        if sel_jobs:
            df_filtered = df[df['job'].isin(sel_jobs)]
            st.write(f"총 **{len(df_filtered)}** 건의 게시글을 분석합니다.")

            # 1) 감성 비율 차트
            st.subheader("감성 분포")
            ratio = df_filtered['pred'].value_counts(normalize=True).reindex(['긍정','중립','부정'], fill_value=0)
            st.bar_chart(ratio)

            # 2) 워드클라우드 (불용어 적용)
            st.subheader("주요 키워드 (워드클라우드)")
            text = " ".join(df_filtered['cleaned'].tolist())
            font_path = "C:/Windows/Fonts/malgun.ttf" # 윈도우 기본 폰트
            try:
                wc = WordCloud(
                    font_path=font_path, width=800, height=400,
                    background_color='white', stopwords=set(stopwords) # 불용어 적용
                ).generate(text)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            except FileNotFoundError:
                st.error(f"폰트 파일을 찾을 수 없습니다: {font_path}")
            except Exception as e:
                st.error(f"워드클라우드 생성 중 오류: {e}")
        else:
            st.warning("분석할 직업을 하나 이상 선택해주세요.")

# --- 탭 2: 실시간 감성 분석 ---
with tab2:
    st.header("실시간 감성 분석기")
    st.write("메이플스토리 관련 텍스트를 입력하면, 학습된 Bi-LSTM 모델이 실시간으로 감성을 분석합니다.")

    if model:
        user_input = st.text_area("분석할 문장을 여기에 입력하세요:", height=150, key="user_input")
        if st.button("분석 실행", key="predict_button"):
            if user_input:
                # 전처리
                tokenized = [word for word in okt.morphs(user_input, stem=True) if word not in stopwords]
                encoded = tokenizer.texts_to_sequences([tokenized])
                padded = pad_sequences(encoded, maxlen=60) # MAX_LEN은 학습과 동일하게
                # 예측
                score = model.predict(padded)
                pred_class = np.argmax(score)
                confidence = score[0][pred_class] * 100
                
                # 결과 출력
                st.subheader("✨ 분석 결과")
                if pred_class == 2: # 긍정
                    st.success(f"긍정적인 의견일 확률이 높습니다. ({confidence:.2f}%)")
                elif pred_class == 1: # 중립
                    st.warning(f"중립적인 의견일 확률이 높습니다. ({confidence:.2f}%)")
                else: # 부정
                    st.error(f"부정적인 의견일 확률이 높습니다. ({confidence:.2f}%)")
                
                st.bar_chart({'긍정': score[0][2], '중립': score[0][1], '부정': score[0][0]})
            else:
                st.warning("분석할 문장을 입력해주세요.")
    else:
        st.error("배포된 분석 모델을 찾을 수 없습니다. 자동화 파이프라인을 먼저 실행해주세요.")
