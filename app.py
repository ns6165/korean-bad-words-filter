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
# 이 파일들이 저장소 최상단에 있어야 합니다.
model = tf.keras.models.load_model('vdcnn_model_with_kogpt2.h5')
with open("tokenizer_with_kogpt2.pickle", "rb") as f:
    tokenizer = pickle.load(f)

maxlen = 1000 

@app.route('/check', methods=['POST'])
def check_toxic():
    try:
        data = request.json
        text = data.get('text', '')
        
        # 2. 직접 변환 (토크나이저 라이브러리 의존성 제거)
        # tokenizer 객체 내 texts_to_sequences가 있으면 사용, 없으면 encode 사용
        if hasattr(tokenizer, 'texts_to_sequences'):
            seq = tokenizer.texts_to_sequences([text.lower()])
        else:
            # 토크나이저가 encode 메서드를 가진 경우
            seq = [tokenizer.encode(text.lower())]
            
        sentence_seq = pad_sequences(seq, maxlen=maxlen, truncating="post")
        
        # 3. 예측
        prediction = model.predict(sentence_seq)[0][0]
        
        # 4. 판별 (필터링 강도 조절: 낮출수록 엄격함)
        return jsonify({'isToxic': bool(prediction >= 0.3)})
        
    except Exception as e:
        print(f"오류 발생: {e}")
        return jsonify({'isToxic': False}) # 서버가 죽지 않게 예외 처리

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
