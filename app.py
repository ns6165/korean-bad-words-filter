from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import pickle
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app)

# 모델 및 토크나이저 로드
try:
    model = tf.keras.models.load_model('vdcnn_model_with_kogpt2.h5')
    with open("tokenizer_with_kogpt2.pickle", "rb") as f:
        tokenizer = pickle.load(f)
    model_loaded = True
except Exception as e:
    print(f"모델 로드 오류: {e}")
    model_loaded = False

@app.route('/check', methods=['POST'])
def check_toxic():
    if not model_loaded:
        return jsonify({'isToxic': False})
    
    try:
        data = request.json
        text = data.get('text', '')
        if not text: return jsonify({'isToxic': False})

        # 토크나이징
        if hasattr(tokenizer, 'texts_to_sequences'):
            seq = tokenizer.texts_to_sequences([text.lower()])
        else:
            seq = [tokenizer.encode(text.lower())]
            
        sentence_seq = pad_sequences(seq, maxlen=1000, truncating="post")
        
        # 모델 예측 (딥러닝 모델의 출력값만 사용)
        prediction = model.predict(sentence_seq)[0][0]
        
        # 로그 확인용 (Render Logs에서 확인 가능)
        print(f"DEBUG: 텍스트='{text}', 예측값={prediction}")
        
        # 0.5 이상이면 True, 아니면 False (임계값 0.5로 복구)
        return jsonify({'isToxic': bool(prediction >= 0.5)})
        
    except Exception as e:
        print(f"예측 오류: {e}")
        return jsonify({'isToxic': False})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
