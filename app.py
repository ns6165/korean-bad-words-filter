from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import pickle
import os
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app)

# 1. 모델과 토크나이저 로드
try:
    model = tf.keras.models.load_model('vdcnn_model_with_kogpt2.h5')
    with open("tokenizer_with_kogpt2.pickle", "rb") as f:
        tokenizer = pickle.load(f)
    model_loaded = True
except Exception as e:
    print(f"로드 오류: {e}")
    model_loaded = False

@app.route('/healthz', methods=['GET'])
def healthz():
    return "OK", 200

@app.route('/check', methods=['POST'])
def check_toxic():
    if not model_loaded: return jsonify({'isToxic': False})
    
    try:
        data = request.json
        text = str(data.get('text', '')).lower()
        if not text: return jsonify({'isToxic': False})
        
        # [핵심] 토크나이징 방식 자동 선택
        # 1. texts_to_sequences를 지원하면 바로 사용
        if hasattr(tokenizer, 'texts_to_sequences'):
            seq = tokenizer.texts_to_sequences([text])
        # 2. encode가 있다면 사용
        elif hasattr(tokenizer, 'encode'):
            seq = [tokenizer.encode(text)]
        # 3. 둘 다 없으면 직접 구현
        else:
            word_index = getattr(tokenizer, 'word_index', {})
            seq = [[word_index.get(w, 0) for w in text.split()]]
            
        sentence_seq = pad_sequences(seq, maxlen=1000, truncating="post")
        
        # 모델 예측
        prediction = model.predict(sentence_seq)[0][0]
        
        print(f"DEBUG_SCORE: 텍스트='{text}', 점수={prediction}")
        
        return jsonify({'isToxic': bool(prediction >= 0.5)})
        
    except Exception as e:
        print(f"최종 해결 시도 에러: {e}")
        return jsonify({'isToxic': False})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
