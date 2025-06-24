# 파일명: training_pipeline.py
import os
import pandas as pd
import numpy as np
import re
import pickle
import shutil
import sys
import datetime
from konlpy.tag import Okt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# --- 설정값 ---
# 1. 일반 게임 데이터 (기본 베이스)
BASE_DATA_PATH = './data/merged_label_final.csv'
# 2. 불용어 사전
STOPWORDS_PATH = './data/korean_stopwords.txt'
# 3. 버전별 모델이 저장될 루트 디렉토리
MODEL_ROOT_DIR = './production_models/'
# 4. 모델 학습 하이퍼파라미터
MAX_LEN = 60
VOCAB_SIZE_LIMIT = 20000
EMBEDDING_DIM = 128
LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 64
DENSE_UNITS = 64
EPOCHS = 50
BATCH_SIZE = 128

def get_latest_model_dir():
    """가장 최신 버전의 모델 디렉토리 경로를 찾습니다."""
    if not os.path.exists(MODEL_ROOT_DIR):
        return None
    # 'v_'로 시작하는 디렉토리만 필터링
    versions = [d for d in os.listdir(MODEL_ROOT_DIR) if os.path.isdir(os.path.join(MODEL_ROOT_DIR, d)) and d.startswith('v_')]
    if not versions:
        return None
    # 이름순으로 정렬하여 가장 마지막 버전(최신)을 반환
    latest_version_dir_name = sorted(versions)[-1]
    return os.path.join(MODEL_ROOT_DIR, latest_version_dir_name)

def load_stopwords(filepath):
    """불용어 사전을 로드합니다."""
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            stopwords = [line.strip() for line in f]
        return stopwords
    print(f"[경고] 불용어 파일 '{filepath}'를 찾을 수 없습니다.")
    return []

def load_and_combine_data(new_labeled_data_path):
    """
    새로운 메이플 데이터와 샘플링된 일반 게임 데이터를 결합하여 학습 데이터셋을 구성합니다.
    """
    print("--- 1. 데이터 로딩 및 결합 시작 ---")
    print(f"신규 메이플 데이터 로딩: {new_labeled_data_path}")
    new_df = pd.read_csv(new_labeled_data_path)
    # 컬럼 이름 통일: 'content', 'label'
    new_df.rename(columns={'sentiment_text': 'content', 'sentiment': 'label'}, inplace=True)

    print(f"기본 일반 게임 데이터 로딩: {BASE_DATA_PATH}")
    base_df = pd.read_csv(BASE_DATA_PATH)

    sample_size = len(new_df)
    print(f"신규 데이터 {sample_size}건에 맞춰 기본 데이터에서 샘플링합니다.")
    if len(base_df) >= sample_size:
        sampled_df = base_df.sample(n=sample_size, random_state=42)
    else:
        # 기본 데이터가 부족하면 있는 만큼만 사용
        sampled_df = base_df
    
    combined_df = pd.concat([new_df, sampled_df], ignore_index=True)
    print(f"결합된 총 학습 데이터 수: {len(combined_df)}건")

    # 데이터 정제
    content_col, label_col = 'content', 'label'
    combined_df.dropna(subset=[content_col, label_col], inplace=True)
    combined_df[content_col] = combined_df[content_col].str.replace(r'[^ㄱ-ㅎㅏ-ㅣ가-힣0-9 ]', '', regex=True).replace('', np.nan)
    combined_df.dropna(subset=[content_col], inplace=True)
    combined_df.drop_duplicates(subset=[content_col], inplace=True)
    combined_df = combined_df[combined_df[label_col].isin([0, 1, 2])]
    
    print(f"최종 전처리 후 데이터 수: {len(combined_df)}")
    return combined_df[content_col].tolist(), combined_df[label_col].astype(int).tolist()

def train_new_model(X_train, y_train, X_val, y_val, stopwords, new_model_path, new_tokenizer_path):
    """주어진 데이터로 새 모델을 학습하고 지정된 경로에 저장합니다."""
    print("\n--- 2. 모델 학습 시작 ---")
    okt = Okt()
    
    print("토큰화 및 불용어 제거 중...")
    X_train_tokenized = [[word for word in okt.morphs(text, stem=True) if word not in stopwords] for text in X_train]
    X_val_tokenized = [[word for word in okt.morphs(text, stem=True) if word not in stopwords] for text in X_val]
    
    tokenizer = Tokenizer(num_words=VOCAB_SIZE_LIMIT, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train_tokenized)
    
    X_train_encoded = tokenizer.texts_to_sequences(X_train_tokenized)
    X_val_encoded = tokenizer.texts_to_sequences(X_val_tokenized)
    
    X_train_padded = pad_sequences(X_train_encoded, maxlen=MAX_LEN)
    X_val_padded = pad_sequences(X_val_encoded, maxlen=MAX_LEN)

    y_train_one_hot = to_categorical(y_train, num_classes=3)
    y_val_one_hot = to_categorical(y_val, num_classes=3)

    model = Sequential([
        Embedding(len(tokenizer.word_index) + 1, EMBEDDING_DIM, input_length=MAX_LEN),
        Dropout(0.3),
        Bidirectional(LSTM(LSTM_UNITS_1, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(LSTM_UNITS_2)),
        Dense(DENSE_UNITS, activation='relu'),
        Dropout(0.4),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1)
    model_checkpoint = ModelCheckpoint(filepath=new_model_path, monitor='val_loss', save_best_only=True, verbose=1)
    
    model.fit(X_train_padded, y_train_one_hot,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              validation_data=(X_val_padded, y_val_one_hot),
              callbacks=[early_stopping, model_checkpoint])
              
    with open(new_tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
        
    print(f"새로운 모델이 '{new_model_path}'에, 토크나이저가 '{new_tokenizer_path}'에 저장되었습니다.")

def evaluate_model(model_path, tokenizer_path, X_test, y_test, stopwords):
    """저장된 모델과 토크나이저로 테스트 데이터셋의 F1-Score를 평가합니다."""
    print(f"\n[평가] '{model_path}'")
    try:
        model = load_model(model_path)
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
    except Exception as e:
        print(f"모델/토크나이저 로딩 실패: {e}")
        return 0.0

    okt = Okt()
    X_test_tokenized = [[word for word in okt.morphs(text, stem=True) if word not in stopwords] for text in X_test]
    X_test_encoded = tokenizer.texts_to_sequences(X_test_tokenized)
    X_test_padded = pad_sequences(X_test_encoded, maxlen=MAX_LEN)
    
    y_pred_probs = model.predict(X_test_padded)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    
    score = f1_score(y_test, y_pred_classes, average='weighted')
    print(f"평가 완료. Weighted F1-Score: {score:.4f}")
    return score

def run_training_pipeline(new_labeled_data_path):
    """학습, 평가, 버전 관리를 총괄하는 메인 파이프라인 함수"""
    stopwords = load_stopwords(STOPWORDS_PATH)
    print(f"불용어 {len(stopwords)}개 로드 완료.")
    
    # 1. 데이터 준비
    X_data, y_data = load_and_combine_data(new_labeled_data_path)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=42, stratify=y_data)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=1/9, random_state=42, stratify=y_train_val)
    
    # 2. 새 모델 학습 및 저장 (버전 관리)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    new_model_version_dir = os.path.join(MODEL_ROOT_DIR, f"v_{timestamp}")
    os.makedirs(new_model_version_dir, exist_ok=True)
    
    new_model_path = os.path.join(new_model_version_dir, 'model.h5')
    new_tokenizer_path = os.path.join(new_model_version_dir, 'tokenizer.pkl')

    train_new_model(X_train, y_train, X_val, y_val, stopwords, new_model_path, new_tokenizer_path)
    
    # 3. 새 모델과 기존 모델 성능 비교
    print("\n--- 3. 모델 성능 비교 ---")
    new_model_score = evaluate_model(new_model_path, new_tokenizer_path, X_test, y_test, stopwords)
    
    latest_prod_model_dir = get_latest_model_dir()
    # 자기 자신(방금 만든 버전)이 아닌, 그 이전의 최신 버전을 찾아야 함
    # 이 로직은 main.py에서 더 정교하게 관리하거나, 여기서 한 단계 전 버전을 찾도록 수정 필요.
    # 여기서는 간단하게 "가장 최신(방금 만든 것 포함)"과 비교하는 것으로 단순화.
    # 실제로는 배포된 모델 목록을 관리하는 것이 좋음.
    if latest_prod_model_dir and os.path.abspath(latest_prod_model_dir) != os.path.abspath(new_model_version_dir):
        old_model_path = os.path.join(latest_prod_model_dir, 'model.h5')
        old_tokenizer_path = os.path.join(latest_prod_model_dir, 'tokenizer.pkl')
        old_model_score = evaluate_model(old_model_path, old_tokenizer_path, X_test, y_test, stopwords)
        
        if new_model_score > old_model_score:
            print(f"\n[결과] 새 모델 성능({new_model_score:.4f})이 기존 모델({old_model_score:.4f})보다 우수합니다.")
        else:
            print(f"\n[결과] 기존 모델 성능이 더 좋거나 같습니다. 새 모델은 '{new_model_version_dir}'에 저장되었지만, 배포 모델은 변경되지 않습니다.")
    else:
        print(f"\n[결과] 비교할 기존 모델이 없습니다. 새 모델이 '{new_model_version_dir}'에 저장되었습니다.")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("사용법: python training_pipeline.py <새로운_라벨링_데이터_경로>")
        print("예시: python training_pipeline.py ./data/labeled_chunks/labeled_data_01.csv")
    else:
        new_labeled_data_file = sys.argv[1]
        run_training_pipeline(new_labeled_data_file)
