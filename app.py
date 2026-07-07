from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import pickle
import os
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app)

# 모델 및 토크나이저 로드
model = tf.keras.models.load_model('vdcnn_model_with_kogpt2.h5')
with open("tokenizer_with_kogpt2.pickle", "rb") as f:
    tokenizer = pickle.load(f)

# [가장 중요한 부분] 라이브러리 메서드 대신 속성(vocab)만 사용
# 만약 word_index가 없다면 학습 파일 구조를 따르는 다른 속성을 찾습니다.
vocab = getattr(tokenizer, 'word_index', getattr(tokenizer, 'vocab', {}))

@app.route('/check', methods=['POST'])
def check_toxic():
    try:
        text = str(request.json.get('text', '')).lower()
        
        # 1. 수동 토큰화 (라이브러리 함수를 사용하지 않고 사전 데이터만 매핑)
        # 단어를 공백으로 쪼개고, 사전에 있으면 해당 ID로, 없으면 0으로 변환
        tokens = text.split()
        seq = [[vocab.get(w, 0) for w in tokens]]
        
        # 2. 패딩 (학습 시 모델이 요구하는 1000차원)
        sentence_seq = pad_sequences(seq, maxlen=1000, truncating="post", padding="post")
        
        # 3. 모델 예측
