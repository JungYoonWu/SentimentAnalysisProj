# 파일명: labeler_main.py
import pandas as pd
import os
import time
from myLangchainService import LLMSentimentAnalyzer

# --- 설정 ---
# 크롤링 결과 원본 파일
RAW_DATA_CSV = './data/posts.csv'
# 라벨링된 결과가 누적될 파일
LABELED_DATA_CSV = './data/labeled_posts.csv'
# LM Studio 설정
SERVER_ENDPOINT = "http://127.0.0.1:1234/v1"
MODEL_NAME = "google/gemma-3-12b"

def get_already_labeled_ids(filename):
    """라벨링된 결과 파일에서 이미 처리된 게시글 번호들을 가져옵니다."""
    if not os.path.exists(filename):
        return set()
    try:
        df = pd.read_csv(filename, dtype={'번호': str})
        return set(df['번호'].tolist())
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return set()

def run_labeling():
    """
    새로운 게시글을 찾아 LLM으로 감성 라벨링을 수행하고 결과를 누적 저장하는 함수.
    """
    # 1. 원본 데이터와 이미 라벨링된 데이터 로드
    if not os.path.exists(RAW_DATA_CSV):
        print(f"원본 데이터 파일 '{RAW_DATA_CSV}'가 없습니다. 크롤링을 먼저 실행하세요.")
        return

    raw_df = pd.read_csv(RAW_DATA_CSV, dtype={'번호': str})
    labeled_ids = get_already_labeled_ids(LABELED_DATA_CSV)
    
    # 2. 라벨링이 필요한 새로운 데이터만 필터링
    new_data_df = raw_df[~raw_df['번호'].isin(labeled_ids)].copy()
    
    if new_data_df.empty:
        print("새롭게 라벨링할 데이터가 없습니다.")
        return
        
    print(f"총 {len(new_data_df)}개의 새로운 데이터에 대해 라벨링을 시작합니다.")

    # 3. LLM 분석기 초기화
    analyzer = LLMSentimentAnalyzer(SERVER_ENDPOINT, MODEL_NAME)
    
    new_labels = []
    total_count = len(new_data_df)

    # 4. 새로운 데이터에 대해 라벨링 수행
    for index, row in new_data_df.iterrows():
        # 제목과 본문을 합쳐서 분석하면 더 정확한 결과를 얻을 수 있습니다.
        # 본문이 비어있는 경우(NaN)를 대비하여 문자열로 변환합니다.
        title = str(row.get('제목', ''))
        body = str(row.get('본문', ''))
        full_text = f"제목: {title}\n본문: {body}"
        
        try:
            # LLM API 호출
            sentiment = analyzer.analyze_sentiment(full_text)
            # '긍정', '부정', '중립' 외의 답변은 '중립'으로 처리
            if sentiment not in ['긍정', '부정', '중립']:
                sentiment = '중립'
        except Exception as e:
            print(f"[오류] LLM API 호출 실패: {e}. 해당 데이터는 '중립'으로 처리합니다.")
            sentiment = '중립'
            time.sleep(5) # API 오류 시 잠시 대기

        new_labels.append(sentiment)
        print(f"[{index+1}/{total_count}] [결과: {sentiment}] {title[:30]}...")
        time.sleep(1) # API 서버 부하 방지를 위한 대기 시간

    new_data_df['sentiment'] = new_labels

    # 5. 라벨링된 새로운 결과를 기존 파일에 추가 (append)
    # 파일이 없으면 헤더와 함께 새로 쓰고, 있으면 헤더 없이 내용만 추가
    header = not os.path.exists(LABELED_DATA_CSV)
    new_data_df.to_csv(LABELED_DATA_CSV, mode='a', header=header, index=False, encoding='utf-8-sig')

    print(f"라벨링 완료. {len(new_data_df)}개의 새로운 결과가 '{LABELED_DATA_CSV}'에 추가되었습니다.")

if __name__ == '__main__':
    run_labeling()
