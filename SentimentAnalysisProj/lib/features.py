# lib/features.py

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def build_tokenizer(texts, num_words=15000):
    tok = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tok.fit_on_texts(texts)
    return tok

def texts_to_padded_sequences(tok, texts, maxlen=80):
    seqs = tok.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=maxlen, padding='pre', truncating='pre')
