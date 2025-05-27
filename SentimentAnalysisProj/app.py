# app.py

import os
import re
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# --- 설정 ---
BASE_DIR  = os.getcwd()
DATA_PATH = os.path.join(BASE_DIR, 'data', 'posts_labeled_pred.csv')

# --- 데이터 불러오기 & 전처리 ---
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    # 제목+본문 텍스트라도 있을 경우 워드클라우드용 텍스트 컬럼 생성
    df['cleaned'] = (
        df['제목'].fillna('') + ' ' + df['본문'].fillna('')
    ).apply(lambda x: re.sub(r"[^가-힣0-9a-zA-Z\s]", " ", str(x)))
    # [직업] 이름만 떼오기
    df['job'] = df['제목'].str.extract(r'^\[(.*?)\]').fillna('Unknown')
    return df

df = load_data()

# --- 사이드바: 게시판(직업군) 선택 ---
st.sidebar.title("🔍 필터")
boards = ['전체'] + sorted(df['게시판'].unique().tolist())
sel_board = st.sidebar.selectbox("직업군(게시판) 선택", boards)

# 선택된 게시판에 속한 직업 리스트 얻기
if sel_board == '전체':
    jobs = sorted(df['job'].unique().tolist())
else:
    jobs = sorted(df[df['게시판'] == sel_board]['job'].unique().tolist())

# --- 사이드바: 직업별 체크박스 (재선택 가능) ---
st.sidebar.markdown("### 직업별 선택")
job_checks = {}
for job in jobs:
    # key 는 job 이름이 유일하다고 가정
    job_checks[job] = st.sidebar.checkbox(job, value=True, key=f"job_{job}")

sel_jobs = [job for job, checked in job_checks.items() if checked]

# --- 데이터 필터링 ---
if sel_board == '전체':
    df_filtered = df[df['job'].isin(sel_jobs)]
else:
    df_filtered = df[
        (df['게시판'] == sel_board) &
        (df['job'].isin(sel_jobs))
    ]

# --- 메인 페이지 출력 ---
st.title("📝 직업별 감성 비율 대시보드")
st.write(f"총 **{len(df_filtered)}** 건의 게시글을 분석합니다.")

# 1) 감성 비율 차트
ratio = (
    df_filtered['pred']
    .value_counts(normalize=True)
    .reindex(['긍정','중립','부정'], fill_value=0)
)
ratio_df = pd.DataFrame({
    '감성': ratio.index,
    '비율': ratio.values
}).set_index('감성')
st.subheader("감성 분포")
st.bar_chart(ratio_df)

# 2) 워드클라우드
st.subheader("워드클라우드")
if not df_filtered.empty:
    text = " ".join(df_filtered['cleaned'].tolist())
    # 한글 폰트 경로를 알맞게 지정해주세요.
    font_path = "C:/Windows/Fonts/malgun.ttf"
    wc = WordCloud(
        font_path=font_path,
        width=800, height=400,
        background_color='white'
    ).generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
else:
    st.write("선택된 조건에 해당하는 데이터가 없습니다.")

# 3) 상세 데이터 테이블
st.subheader("상세 데이터")
st.dataframe(
    df_filtered[['게시판','job','제목','pred']].reset_index(drop=True),
    use_container_width=True
)
