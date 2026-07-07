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

        # 토크나이징 (Tokenizer가 가진 방식대로 숫자 배열로 변환)
        if hasattr(tokenizer, 'texts_to_sequences'):
            seq = tokenizer.texts_to_sequences([text.lower()])
        else:
            seq = [tokenizer.encode(text.lower())]
            
        sentence_seq = pad_sequences(seq, maxlen=1000, truncating="post")
        
        # 모델 예측
        prediction = model.predict(sentence_seq)[0][0]
        
        # [로그 확인] 이 부분이 Render의 Logs 탭에 실시간으로 찍힙니다.
        # 욕설 입력 시 예측값이 어떻게 나오는지 여기서 확인 가능합니다.
        print(f"DEBUG_LOG: 입력텍스트='{text}', 모델예측점수={prediction}")
        
        # 임계값 0.5 이상이면 True로 판정
        return jsonify({'isToxic': bool(prediction >= 0.5)})
        
    except Exception as e:
        print(f"예측 중 오류 발생: {e}")
        return jsonify({'isToxic': False})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
