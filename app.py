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
    print(f"모델 로드 오류: {e}")
    model_loaded = False

@app.route('/healthz', methods=['GET'])
def healthz():
    return "OK", 200

@app.route('/check', methods=['POST'])
def check_toxic():
    if not model_loaded:
        return jsonify({'isToxic': False})
    
    try:
        data = request.json
        text = data.get('text', '')
        if not text: return jsonify({'isToxic': False})

        # 2. 토크나이징 (Tokenizer가 가진 기능만 최소한으로 호출)
        # hasattr 체크를 빼고 바로 시도합니다.
        try:
            # 1순위: texts_to_sequences 시도
            seq = tokenizer.texts_to_sequences([text.lower()])
        except:
            # 2순위: encode 시도
            seq = [tokenizer.encode(text.lower())]
            
        sentence_seq = pad_sequences(seq, maxlen=1000, truncating="post")
        
        # 3. 예측
        prediction = model.predict(sentence_seq)[0][0]
        
        # [로그 확인] 이 값을 보면 왜 안 걸러지는지 바로 나옵니다.
        print(f"DEBUG_SCORE: 텍스트='{text}', 예측점수={prediction}")
        
        # 임계값을 낮춰서라도 욕설을 잡도록 설정
        return jsonify({'isToxic': bool(prediction >= 0.1)})
        
    except Exception as e:
        print(f"예측 에러: {e}")
        return jsonify({'isToxic': False})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
