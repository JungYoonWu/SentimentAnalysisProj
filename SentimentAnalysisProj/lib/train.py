import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense

from lib.preprocessing import clean_text, tokenize
from lib.features import build_tokenizer, texts_to_padded_sequences

def main():
    # 디렉터리 설정
    base_dir  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir  = os.path.join(base_dir, 'data')
    model_dir = os.path.join(base_dir, 'model')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 1) 샘플 데이터로 학습 (posts_sample.csv)
    sample_path = os.path.join(data_dir, 'posts_sample.csv')
    df_sample = pd.read_csv(sample_path, encoding='cp949')
    if 'label' not in df_sample.columns:
        raise RuntimeError("posts_sample.csv에 'label' 컬럼이 없습니다.")

    # 텍스트 전처리
    df_sample['text']    = (df_sample['제목'].fillna('') + ' ' + df_sample['본문'].fillna('')).apply(clean_text)
    df_sample['tokens']  = df_sample['text'].apply(tokenize)
    df_sample['cleaned'] = df_sample['tokens'].apply(lambda t: " ".join(t))

    # 레이블 인코딩 & one-hot
    le     = LabelEncoder()
    y_idx  = le.fit_transform(df_sample['label'])
    y      = np.eye(len(le.classes_))[y_idx]

    # train/val 분할
    X_train, X_val, y_train, y_val = train_test_split(
        df_sample['cleaned'], y,
        test_size=0.2,
        stratify=df_sample['label'],
        random_state=42
    )

    # 토크나이저 생성 & 패딩
    tokenizer    = build_tokenizer(X_train)
    X_train_pad   = texts_to_padded_sequences(tokenizer, X_train)
    X_val_pad     = texts_to_padded_sequences(tokenizer, X_val)

    # 모델 정의/컴파일/학습
    model = Sequential([
        Embedding(input_dim=15000, output_dim=128, input_length=80),
        LSTM(128),
        Dropout(0.5),
        Dense(len(le.classes_), activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(
        X_train_pad, y_train,
        validation_data=(X_val_pad, y_val),
        epochs=5, batch_size=32
    )

    # 모델·토크나이저·레이블인코더 저장
    model_path = os.path.join(model_dir, '20250520_model.h5')
    model.save(model_path)
    with open(os.path.join(model_dir, 'tokenizer.pkl'), 'wb') as f:
        pickle.dump(tokenizer, f)
    with open(os.path.join(model_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(le, f)
    print(f"✅ 모델 저장 완료: {model_path}")

    # 2) 학습된 모델로 나머지 데이터 라벨링
    from lib.inference import predict_texts

    def label_df(input_name, output_name):
        path_in  = os.path.join(data_dir, input_name)
        path_out = os.path.join(data_dir, output_name)
        df = pd.read_csv(path_in, encoding='cp949')
        df['text'] = (df.get('제목','').fillna('') + ' ' + df.get('본문','').fillna('')).apply(clean_text)
        preds = predict_texts(df['text'].tolist())
        df['label'] = [lbl for _, lbl, _ in preds]
        df.to_csv(path_out, index=False, encoding='cp949')
        print(f"✅ 라벨링 완료: {output_name}")

    # postsTrain.csv → postsTrain_labeled.csv
    label_df('postsTrain.csv', 'postsTrain_labeled.csv')
    # posts.csv      → posts_labeled.csv
    label_df('posts.csv',      'posts_labeled.csv')

if __name__ == '__main__':
    main()
