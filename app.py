# 파일명: app.py
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
import glob

# --- 앱 기본 설정 (가장 먼저 실행) ---
st.set_page_config(layout="wide", page_title="메이플 여론 분석 대시보드")

# --- 경로 설정 ---
BASE_DIR = os.getcwd()
MODEL_ROOT_DIR = os.path.join(BASE_DIR, 'production_models')
LABELED_CHUNK_DIR = os.path.join(BASE_DIR, 'data', 'labeled_chunks')
STOPWORDS_PATH = os.path.join(BASE_DIR, 'data', 'korean_stopwords.txt')
# ★ 사용자 피드백 저장 경로 추가
FEEDBACK_FILE_PATH = os.path.join(BASE_DIR, 'data', 'user_feedback.csv')
FONT_PATH = "C:/Windows/Fonts/malgun.ttf"
MAX_LEN = 60

# --- 데이터 및 모델 로딩 (캐싱 기능으로 성능 최적화) ---
@st.cache_resource
def load_prediction_assets():
    """실시간 예측에 필요한 최신 버전의 모델, 토크나이저, Okt, 불용어를 로드합니다."""
    # ... (이전과 동일)
    if not os.path.exists(MODEL_ROOT_DIR): return None, None, None, [], "모델 루트 디렉토리가 없습니다."
    version_dirs = glob.glob(os.path.join(MODEL_ROOT_DIR, 'v_*'))
    if not version_dirs: return None, None, None, [], "학습된 모델 버전이 없습니다."
    latest_version_dir = max(version_dirs, key=os.path.getctime)
    model_path = os.path.join(latest_version_dir, 'model.h5')
    tokenizer_path = os.path.join(latest_version_dir, 'tokenizer.pkl')
    try:
        _model = load_model(model_path)
        with open(tokenizer_path, 'rb') as f: _tokenizer = pickle.load(f)
        with open(STOPWORDS_PATH, 'r', encoding='utf-8') as f: _stopwords = [line.strip() for line in f]
        _okt = Okt()
        return _model, _tokenizer, _okt, _stopwords, f"최신 모델 '{os.path.basename(latest_version_dir)}' 로드 완료."
    except Exception as e:
        return None, None, None, [], f"모델 로딩 중 오류: {e}"

@st.cache_data
def load_dashboard_data():
    """
    대시보드 시각화에 사용할 가장 최신 라벨링 데이터를 로드하고 전처리합니다.
    ★ 디버깅 정보 출력을 추가했습니다.
    """
    st.subheader("🕵️‍♂️ 대시보드 데이터 로딩 과정 진단")
    
    if not os.path.exists(LABELED_CHUNK_DIR):
        return pd.DataFrame(), f"오류: 라벨링된 데이터 폴더 '{LABELED_CHUNK_DIR}'가 없습니다."

    labeled_files = glob.glob(os.path.join(LABELED_CHUNK_DIR, 'labeled_data_*.csv'))
    if not labeled_files:
        return pd.DataFrame(), f"오류: '{LABELED_CHUNK_DIR}' 폴더에 라벨링된 데이터 파일이 없습니다."

    latest_data_path = max(labeled_files, key=os.path.getctime)
    st.info(f"1. 최신 데이터 파일 확인: '{os.path.basename(latest_data_path)}'")
    
    try:
        df = pd.read_csv(latest_data_path, encoding='utf-8-sig')
        st.info(f"2. 파일 로드 성공: 총 {len(df)}개의 행을 읽었습니다.")
        st.write("3. 원본 데이터 컬럼 목록:", df.columns.tolist())

        # 필수 컬럼 확인
        required_cols = ['제목', '본문', 'label']
        if not all(col in df.columns for col in required_cols):
             return pd.DataFrame(), f"오류: 파일에 필수 컬럼({required_cols}) 중 일부가 없습니다."

        # 전처리
        df.dropna(subset=['label'], inplace=True)
        st.info(f"4. 'label'이 없는 행 제거 후: {len(df)}개 행 남음")

        df['cleaned_text'] = (df['제목'].fillna('') + ' ' + df['본문'].fillna('')).apply(lambda x: re.sub(r"[^가-힣0-9a-zA-Z\s]", " ", str(x)))
        df['job'] = df['제목'].str.extract(r'^\[(.*?)\]').fillna('기타')
        df['sentiment_name'] = df['label'].map({0.0: '부정', 1.0: '중립', 2.0: '긍정'}) # label이 float일 경우 대비
        
        st.success("5. 전처리 완료. 대시보드를 표시합니다.")
        return df, f"최신 데이터 '{os.path.basename(latest_data_path)}' 로드 완료."
    except Exception as e:
        return pd.DataFrame(), f"오류: 데이터 처리 중 예외 발생 - {e}"

# --- 앱 실행 및 UI 구성 ---
st.title("🍁 메이플스토리 직업별 여론 대시보드 & 실시간 감성 분석")

# 데이터 및 모델 로드
df, dashboard_msg = load_dashboard_data()
model, tokenizer, okt, stopwords, model_msg = load_prediction_assets()

# 탭 생성
tab1, tab2 = st.tabs(["📊 대시보드", "🤖 실시간 감성 분석"])

with tab1:
    st.info(dashboard_msg)
    if df.empty:
        # 이 메시지가 보인다면, load_dashboard_data 함수가 빈 데이터프레임을 반환한 것
        st.error("대시보드에 표시할 데이터가 없습니다. 위의 진단 메시지를 확인해주세요.")
    else:
        # (이하 대시보드 UI 로직은 이전과 거의 동일)
        col1, col2 = st.columns([1, 3])
        with col1:
            st.header("🔍 필터")
            boards = ['전체'] + sorted(df['게시판'].unique().tolist())
            sel_board = st.selectbox("게시판 선택", boards, key='board_select')
            if sel_board == '전체':
                jobs = sorted(df['job'].unique().tolist())
            else:
                jobs = sorted(df[df['게시판'] == sel_board]['job'].unique().tolist())
            sel_jobs = st.multiselect("직업 선택", jobs, default=jobs, key='job_multiselect')
        with col2:
            if sel_jobs:
                df_filtered = df[df['job'].isin(sel_jobs)]
                st.write(f"총 **{len(df_filtered)}** 건의 게시글을 분석합니다.")
                st.subheader("감성 분포")
                # 'sentiment_name' 컬럼이 없는 경우 대비
                if 'sentiment_name' in df_filtered.columns:
                    ratio = df_filtered['sentiment_name'].value_counts(normalize=True).reindex(['긍정', '중립', '부정'], fill_value=0)
                    st.bar_chart(ratio)
                else:
                    st.warning("'sentiment_name' 컬럼을 찾을 수 없습니다.")

                st.subheader("주요 키워드 (워드클라우드)")
                text = " ".join(df_filtered['cleaned_text'].dropna().tolist())
                if text:
                    try:
                        wc = WordCloud(font_path=FONT_PATH, width=800, height=400, background_color='white', stopwords=set(stopwords)).generate(text)
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.imshow(wc, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"워드클라우드 생성 중 오류: {e}")
            else:
                st.warning("분석할 직업을 선택해주세요.")

with tab2:
    st.info(model_msg)
    st.header("실시간 감성 분석기")
    if model and tokenizer:
        user_input = st.text_area("분석할 문장을 여기에 입력하세요:", height=150, key="user_input")
        if st.button("분석 실행", key="predict_button"):
            if user_input.strip():
                with st.spinner('모델이 감성을 분석하고 있습니다...'):
                    # 모델 예측 로직
                    tokenized = [word for word in okt.morphs(user_input, stem=True) if word not in stopwords]
                    encoded = tokenizer.texts_to_sequences([tokenized])
                    padded = pad_sequences(encoded, maxlen=MAX_LEN)
                    score = model.predict(padded)
                    pred_class = np.argmax(score, axis=1)[0]
                    # 예측 결과를 세션 상태에 저장 (피드백 저장을 위해)
                    st.session_state['last_input'] = user_input
                    st.session_state['pred_class'] = pred_class
                    st.session_state['score'] = score
            else:
                st.warning("분석할 문장을 입력해주세요.")

        # 세션 상태에 분석 결과가 있을 경우에만 결과 및 피드백 UI 표시
        if 'last_input' in st.session_state:
            pred_class = st.session_state['pred_class']
            score = st.session_state['score']
            confidence = score[0][pred_class] * 100
            
            st.subheader("✨ 분석 결과")
            if pred_class == 2: st.success(f"**긍정적인 의견**일 확률이 높습니다. ({confidence:.2f}%)")
            elif pred_class == 1: st.warning(f"**중립적인 의견**일 확률이 높습니다. ({confidence:.2f}%)")
            else: st.error(f"**부정적인 의견**일 확률이 높습니다. ({confidence:.2f}%)")
            
            st.bar_chart({'긍정': score[0][2], '중립': score[0][1], '부정': score[0][0]})
            
            # ★ 사용자 피드백 UI 부분 ★
            st.write("---")
            st.subheader("✍️ 모델의 판단이 정확한가요? 피드백을 남겨주세요.")
            
            with st.form(key='feedback_form'):
                sentiment_options = ['긍정', '중립', '부정']
                # 모델의 예측을 기본값으로 설정
                default_index = {2: 0, 1: 1, 0: 2}.get(pred_class, 1) # 긍정(2)->0, 중립(1)->1, 부정(0)->2
                
                user_label_text = st.radio(
                    label="이 문장의 실제 감성은 무엇인가요?",
                    options=sentiment_options,
                    index=default_index,
                    horizontal=True
                )
                submitted = st.form_submit_button("피드백 저장하기")
                
                if submitted:
                    label_map_to_int = {'긍정': 2, '중립': 1, '부정': 0}
                    final_label = label_map_to_int[user_label_text]
                    text_to_save = st.session_state['last_input']
                    
                    if save_feedback(text_to_save, final_label):
                        st.success("소중한 피드백 감사합니다! 데이터가 성공적으로 저장되었습니다.")
                        # 피드백 저장 후 세션 상태 초기화
                        del st.session_state['last_input']
                        del st.session_state['pred_class']
                        del st.session_state['score']
                        # st.experimental_rerun() # 필요시 화면 새로고침
    else:
        st.error("실시간 분석에 필요한 모델을 로드할 수 없습니다.")
