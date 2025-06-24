# lib/inference.py

import os
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from preprocessing import clean_text, tokenize
from features import texts_to_padded_sequences

# 프로젝트 루트 기준 model 디렉터리 설정
_base_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_model_dir  = os.path.join(_base_dir, 'model')

_model      = load_model(os.path.join(_model_dir, '20250520_model.h5'))
with open(os.path.join(_model_dir, 'tokenizer.pkl'), 'rb') as f:
    _tokenizer = pickle.load(f)
with open(os.path.join(_model_dir, 'label_encoder.pkl'), 'rb') as f:
    _label_encoder = pickle.load(f)

def predict_texts(texts: list[str]) -> list[tuple[str, str, list[float]]]:
    cleaned = [clean_text(t) for t in texts]
    joined  = [" ".join(tokenize(t)) for t in cleaned]
    seqs    = texts_to_padded_sequences(_tokenizer, joined)
    probs   = _model.predict(seqs)
    labels  = _label_encoder.inverse_transform(np.argmax(probs, axis=1))
    return list(zip(texts, labels, probs.tolist()))
