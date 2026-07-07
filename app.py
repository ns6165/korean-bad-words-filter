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
@app.route('/healthz', methods=['GET'])
def health():
    return '', 200
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
        prediction = model.predict(sentence_seq)
        is_toxic = bool(prediction[0][0] > 0.5) # 모델 결과에 따른 판단
        
        return jsonify({'isToxic': is_toxic})

    except Exception as e:
        # 에러가 발생해도 서버가 죽지 않고 클라이언트에게 알려줌
        print(f"Error occurred: {e}")
        return jsonify({'isToxic': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
