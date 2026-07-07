from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import pickle
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app)

# 1. 모델과 토크나이저 로드 (학습 때 사용한 방식 그대로)
model = tf.keras.models.load_model('vdcnn_model_with_kogpt2.h5')
with open("tokenizer_with_kogpt2.pickle", "rb") as f:
    tokenizer = pickle.load(f)

@app.route('/check', methods=['POST'])
def check_toxic():
    try:
        data = request.json
        text = data.get('text', '').lower()
        
        # 2. 예시 코드의 encode_plus 로직 (학습 때 사용한 방식)
        # 이 토크나이저는 반드시 encode_plus를 가져야 합니다.
        encoded_sentence = tokenizer.encode_plus(
            text,
            max_length=1000,
            padding="max_length",
            truncation=True
        )['input_ids']
        
        # 3. 모델 추론
        sentence_seq = pad_sequences([encoded_sentence], maxlen=1000, truncating="post")
        prediction = model.predict(sentence_seq)[0][0]
        
        # 4. 판정
        return jsonify({'isToxic': bool(prediction >= 0.5)})
    except Exception as e:
        return jsonify({'isToxic': False})
