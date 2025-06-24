# 파일명: inference_app.py
import streamlit as st
import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from konlpy.tag import Okt

# --- 설정: 배포된 모델과 토크나이저 경로 ---
PROD_MODEL_DIR = './production_model/'
PROD_MODEL_PATH = os.path.join(PROD_MODEL_DIR, 'best_maple_model.h5')
PROD_TOKENIZER_PATH = os.path.join(PROD_MODEL_DIR, 'tokenizer.pkl')
MAX_LEN = 60 # training_pipeline.py와 동일한 값

# --- 모델과 토크나이저 로딩 (메모리에 캐싱하여 재실행 속도 향상) ---
@st.cache_resource
def load_assets():
    """배포된 모델과 토크나이저를 로드합니다."""
    if not os.path.exists(PROD_MODEL_PATH) or not os.path.exists(PROD_TOKENIZER_PATH):
        return None, None, None
    try:
        model = load_model(PROD_MODEL_PATH)
        with open(PROD_TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        okt = Okt()
        return model, tokenizer, okt
    except Exception as e:
        st.error(f"모델 로딩 중 오류 발생: {e}")
        return None, None, None

model, tokenizer, okt = load_assets()

# --- Streamlit UI 구성 ---
st.set_page_config(page_title="메이플 인벤 감성 분석기", page_icon="🍁")
st.title("🍁 메이플 인벤 게시글 감성 분석기")
st.write("---")
st.write("메이플스토리 커뮤니티 게시글의 긍정/중립/부정 어조를 분석합니다.")

if model is None:
    st.error("배포된 모델을 찾을 수 없습니다. `main.py` 파이프라인을 먼저 실행하여 모델을 생성해주세요.")
else:
    # 사용자 입력
    user_input = st.text_area("분석할 문장을 입력하세요:", "이 게임 정말 재밌네요! 캐릭터도 귀엽고...", height=150)

    if st.button("감성 분석 실행하기"):
        if user_input:
            # 1. 전처리 (형태소 분석, 정수 인코딩, 패딩)
            tokenized_sentence = okt.morphs(user_input, stem=True)
            encoded_sentence = tokenizer.texts_to_sequences([tokenized_sentence])
            padded_sentence = pad_sequences(encoded_sentence, maxlen=MAX_LEN)

            # 2. 예측
            score = model.predict(padded_sentence)
            prediction = np.argmax(score)
            confidence = score[0][prediction] * 100

            # 3. 결과 표시
            st.write("---")
            st.subheader("📊 분석 결과")
            
            if prediction == 0:
                st.error(f"**부정적인 의견**일 확률이 높습니다. ({confidence:.2f}%)")
            elif prediction == 1:
                st.warning(f"**중립적인 의견**일 확률이 높습니다. ({confidence:.2f}%)")
            else: # prediction == 2
                st.success(f"**긍정적인 의견**일 확률이 높습니다. ({confidence:.2f}%)")

            # 각 클래스별 확률 시각화
            st.write("세부 확률:")
            chart_data = {'긍정': score[0][2], '중립': score[0][1], '부정': score[0][0]}
            st.bar_chart(chart_data)
        else:
            st.warning("분석할 문장을 입력해주세요.")

