from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import pickle
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app)

# 모델 로드 (에러가 나면 서버가 실행조차 안 되므로, 로드 실패 시 예외 처리 필요)
try:
    model = tf.keras.models.load_model('vdcnn_model_with_kogpt2.h5')
    with open("tokenizer_with_kogpt2.pickle", "rb") as f:
        tokenizer = pickle.load(f)
    model_loaded = True
except Exception as e:
    print(f"모델 로드 실패: {e}")
    model_loaded = False

maxlen = 1000

@app.route('/check', methods=['POST'])
def check_toxic():
    if not model_loaded:
        return jsonify({'isToxic': False, 'error': 'Model not loaded'})

    try:
        data = request.json
        text = data.get('text', '')
        if not text:
            return jsonify({'isToxic': False})

        # 토크나이저 예측 수행
        encoded = tokenizer.encode(text.lower())
        # 리스트 형태가 아닐 경우 처리
        if isinstance(encoded, dict):
            encoded = encoded.get('input_ids', [])
        
        sentence_seq = pad_sequences([encoded], maxlen=maxlen, truncating="post")
       prediction = model.predict(sentence_seq)[0][0]
print(f"DEBUG: Input: {text}, Score: {prediction}")
        
        return jsonify({'isToxic': bool(prediction >= 0.5)})
        
    except Exception as e:
        print(f"서버 내부 오류: {e}")
        # 에러가 나도 서버가 죽지 않고 false를 리턴하게 함 (안전성 확보)
        return jsonify({'isToxic': False})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
