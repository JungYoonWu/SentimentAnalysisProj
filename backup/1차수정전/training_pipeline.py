# 파일명: training_pipeline.py
import os
import pandas as pd
import numpy as np
import re
import pickle
import shutil
from konlpy.tag import Okt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# --- 설정값 ---
LABELED_DATA_PATH = './data/merged_label_final.csv'
STOPWORDS_PATH = './data/korean_stopwords.txt'

PROD_MODEL_DIR = './production_model/'
NEW_MODEL_DIR = './new_model/'
PROD_MODEL_PATH = os.path.join(PROD_MODEL_DIR, 'best_maple_model.h5')
PROD_TOKENIZER_PATH = os.path.join(PROD_MODEL_DIR, 'tokenizer.pkl')
NEW_MODEL_PATH = os.path.join(NEW_MODEL_DIR, 'best_maple_model.h5')
NEW_TOKENIZER_PATH = os.path.join(NEW_MODEL_DIR, 'tokenizer.pkl')

MAX_LEN = 60
VOCAB_SIZE_LIMIT = 20000

def load_stopwords(filepath):
    """불용어 사전을 로드합니다."""
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            stopwords = [line.strip() for line in f]
        return stopwords
    return []

def load_and_preprocess_data(file_path):
    """라벨링된 CSV 파일을 로드하고 전처리합니다."""
    print(f"데이터 로딩: {file_path}")
    df = pd.read_csv(file_path)
    content_col, label_col = 'content', 'label'
    
    df.dropna(subset=[content_col, label_col], inplace=True)
    df[content_col] = df[content_col].str.replace(r'[^ㄱ-ㅎㅏ-ㅣ가-힣0-9 ]', '', regex=True)
    df.dropna(subset=[content_col], inplace=True)
    df.drop_duplicates(subset=[content_col], inplace=True)
    df = df[df[label_col].isin([0, 1, 2])]
    
    X_data = df[content_col].tolist()
    y_data = df[label_col].astype(int).tolist()
    
    print(f"전처리 후 데이터 수: {len(X_data)}")
    return X_data, y_data

def train_new_model(X_train, y_train, X_val, y_val, stopwords):
    """새로운 데이터로 모델을 학습하고 저장합니다."""
    print("\n[학습 시작] 새로운 모델 학습을 시작합니다.")
    okt = Okt()
    
    # 1. 토큰화 및 불용어 제거
    print("토큰화 및 불용어 제거 중...")
    X_train_tokenized = [[word for word in okt.morphs(text, stem=True) if word not in stopwords] for text in X_train]
    X_val_tokenized = [[word for word in okt.morphs(text, stem=True) if word not in stopwords] for text in X_val]
    
    tokenizer = Tokenizer(num_words=VOCAB_SIZE_LIMIT)
    tokenizer.fit_on_texts(X_train_tokenized)
    
    X_train_encoded = tokenizer.texts_to_sequences(X_train_tokenized)
    X_val_encoded = tokenizer.texts_to_sequences(X_val_tokenized)
    
    X_train_padded = pad_sequences(X_train_encoded, maxlen=MAX_LEN)
    X_val_padded = pad_sequences(X_val_encoded, maxlen=MAX_LEN)

    y_train_one_hot = to_categorical(y_train, num_classes=3)
    y_val_one_hot = to_categorical(y_val, num_classes=3)

    model = Sequential([
        Embedding(len(tokenizer.word_index) + 1, 128, input_length=MAX_LEN),
        Bidirectional(LSTM(128)),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    os.makedirs(NEW_MODEL_DIR, exist_ok=True)
    model_checkpoint = ModelCheckpoint(filepath=NEW_MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1)
    
    model.fit(X_train_padded, y_train_one_hot,
              epochs=10, batch_size=128,
              validation_data=(X_val_padded, y_val_one_hot),
              callbacks=[early_stopping, model_checkpoint])
              
    with open(NEW_TOKENIZER_PATH, 'wb') as f:
        pickle.dump(tokenizer, f)
        
    print(f"새로운 모델과 토크나이저가 '{NEW_MODEL_DIR}'에 저장되었습니다.")
    return True

def evaluate_model(model_path, tokenizer_path, X_test, y_test, stopwords):
    """저장된 모델과 토크나이저로 테스트 데이터셋을 평가합니다."""
    print(f"\n[평가] '{model_path}' 모델 성능을 평가합니다.")
    try:
        model = load_model(model_path)
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
    except Exception as e:
        print(f"모델 또는 토크나이저 로딩 실패: {e}")
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

def run_training_pipeline():
    """학습, 평가, 배포를 총괄하는 메인 파이프라인 함수"""
    stopwords = load_stopwords(STOPWORDS_PATH)
    print(f"불용어 {len(stopwords)}개 로드 완료.")
    
    X_data, y_data = load_and_preprocess_data(LABELED_DATA_PATH)
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=42, stratify=y_data)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=1/9, random_state=42, stratify=y_train_val)
    
    print(f"학습 데이터: {len(X_train)}개, 검증 데이터: {len(X_val)}개, 테스트 데이터: {len(X_test)}개")
    
    train_new_model(X_train, y_train, X_val, y_val, stopwords)
    
    new_model_score = evaluate_model(NEW_MODEL_PATH, NEW_TOKENIZER_PATH, X_test, y_test, stopwords)
    
    if os.path.exists(PROD_MODEL_PATH):
        old_model_score = evaluate_model(PROD_MODEL_PATH, PROD_TOKENIZER_PATH, X_test, y_test, stopwords)
        
        if new_model_score > old_model_score:
            print(f"\n[배포] 새 모델 성능({new_model_score:.4f})이 기존 모델({old_model_score:.4f})보다 우수하여 모델을 교체합니다.")
            os.makedirs(PROD_MODEL_DIR, exist_ok=True)
            shutil.copy(NEW_MODEL_PATH, PROD_MODEL_PATH)
            shutil.copy(NEW_TOKENIZER_PATH, PROD_TOKENIZER_PATH)
        else:
            print(f"\n[배포 보류] 기존 모델 성능이 더 좋거나 같으므로 교체하지 않습니다.")
    else:
        print(f"\n[최초 배포] 기존 모델이 없으므로 새 모델을 배포합니다.")
        os.makedirs(PROD_MODEL_DIR, exist_ok=True)
        shutil.copy(NEW_MODEL_PATH, PROD_MODEL_PATH)
        shutil.copy(NEW_TOKENIZER_PATH, PROD_TOKENIZER_PATH)

if __name__ == '__main__':
    run_training_pipeline()
