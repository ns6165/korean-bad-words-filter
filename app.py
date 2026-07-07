from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import pickle
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app)

# 모델 로드 (에러 방지용)
try:
    model = tf.keras.models.load_model('vdcnn_model_with_kogpt2.h5')
    with open("tokenizer_with_kogpt2.pickle", "rb") as f:
        tokenizer = pickle.load(f)
    model_loaded = True
except Exception as e:
    print(f"모델 로드 오류: {e}")
    model_loaded = False

# [핵심] Render의 Health Check용 경로 추가 (서버 죽음 방지)
@app.route('/healthz', methods=['GET'])
def healthz():
    return "OK", 200

@app.route('/check', methods=['POST'])
def check_toxic():
    if not model_loaded: return jsonify({'isToxic': False})
    
    try:
        data = request.json
        text = data.get('text', '')
        
        # 텍스트 변환 및 모델 예측
        if hasattr(tokenizer, 'texts_to_sequences'):
            seq = tokenizer.texts_to_sequences([text.lower()])
        else:
            seq = [tokenizer.encode(text.lower())]
            
        sentence_seq = pad_sequences(seq, maxlen=1000, truncating="post")
prediction = model.predict(sentence_seq)[0][0]
print(f"DEBUG: 텍스트='{text}', 예측값={prediction}") # 예측값이 몇인지 로그에서 확인 가능
is_toxic = bool(prediction >= 0.1)        
        return jsonify({'isToxic': bool(prediction >= 0.3)})
    except Exception as e:
        return jsonify({'isToxic': False})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
