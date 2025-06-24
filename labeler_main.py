# 파일명: labeler_main.py
import pandas as pd
import os
import time
import sys
from myLangchainService import LLMSentimentAnalyzer

# --- 설정 ---
SERVER_ENDPOINT = "http://127.0.0.1:1234/v1"
MODEL_NAME = "google/gemma-3-12b"

def run_labeling(input_path, output_path):
    """
    주어진 CSV 파일의 'content' 컬럼을 라벨링하여 새로운 CSV 파일로 저장합니다.
    """
    if not os.path.exists(input_path):
        print(f"입력 파일 '{input_path}'를 찾을 수 없습니다.")
        return

    df = pd.read_csv(input_path)
    # 제목과 본문을 합친 content 컬럼 생성
    df['content'] = df['제목'].fillna('') + '\n' + df['본문'].fillna('')
    
    print(f"총 {len(df)}개의 데이터에 대해 라벨링을 시작합니다.")

    analyzer = LLMSentimentAnalyzer(SERVER_ENDPOINT, MODEL_NAME)
    labels = []
    
    for index, row in df.iterrows():
        try:
            sentiment = analyzer.analyze_sentiment(row['content'])
            if sentiment not in ['긍정', '부정', '중립']:
                sentiment = '중립'
            labels.append(sentiment)
        except Exception as e:
            print(f"[오류] API 호출 실패: {e}. '중립'으로 처리합니다.")
            labels.append('중립')
            time.sleep(5)

        print(f"[{index + 1}/{len(df)}] [결과: {sentiment}] {str(row['제목'])[:30]}...")
        time.sleep(1)
        
    df['sentiment_text'] = labels
    
    # 텍스트 라벨을 숫자로 변환 (0: 부정, 1: 중립, 2: 긍정)
    label_map = {'부정': 0, '중립': 1, '긍정': 2}
    df['label'] = df['sentiment_text'].map(label_map)
    
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"라벨링 완료. 결과가 '{output_path}'에 저장되었습니다.")

if __name__ == '__main__':
    # 스크립트를 직접 실행할 때 명령줄 인수를 받도록 설정
    if len(sys.argv) != 3:
        print("사용법: python labeler_main.py <입력_파일_경로> <출력_파일_경로>")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        run_labeling(input_file, output_file)
