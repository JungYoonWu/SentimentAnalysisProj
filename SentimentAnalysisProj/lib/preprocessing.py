# lib/preprocessing.py

import os
import re
from konlpy.tag import Okt

okt = Okt()
# stopwords 파일 경로를 lib 폴더 기준으로 설정
_stopwords_path = os.path.join(os.path.dirname(__file__), 'korean_stopwords.txt')
stopwords = set(open(_stopwords_path, 'r', encoding='utf-8').read().split())

def clean_text(text: str) -> str:
    text = re.sub(r"[^가-힣0-9a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def tokenize(text: str) -> list[str]:
    tokens = okt.pos(clean_text(text), norm=True, stem=True)
    return [
        w for w, p in tokens
        if p in ('Noun','Verb','Adjective') and w not in stopwords
    ]
