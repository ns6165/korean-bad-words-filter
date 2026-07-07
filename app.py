from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import pickle
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app)

# 1. 모델과 토크나이저 로드 (제시해주신 코드 구조 그대로)
maxlen = 1000  # 예시 코드의 200이 아닌, 실제 모델 학습 시 1000으로 하셨던 것으로 유지
model_path = 'vdcnn_model_with_kogpt2.h5'
tokenizer_path = "tokenizer_with_kogpt2.pickle"

model = tf.keras.models.load_model(model_path)
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)

# 2. 전처리 및 예측 함수 (선생님 예시 로직 통합)
def preprocess_text(text):
    return text.lower()

@app.route('/healthz', methods=['GET'])
def healthz():
    return "OK", 200

@app.route('/check', methods=['POST'])
def check_toxic():
    try:
        data = request.json
        text = data.get('text', '')
        
        sentence = preprocess_text(text)
        
        # [예시 코드 방식 적용] encode_plus 활용
        # tokenizer 객체가 encode_plus를 가지고 있는지 확인하고 사용
        if hasattr(tokenizer, 'encode_plus'):
            encoded = tokenizer.encode_plus(
                sentence,
                max_length=maxlen,
                padding="max_length",
                truncation=True
            )['input_ids']
            sentence_seq = pad_sequences([encoded], maxlen=maxlen, truncating="post")
        else:
            # 혹시 모를 상황 대비 (학습된 토크나이저가 다른 경우)
            seq = tokenizer.texts_to_sequences([sentence])
            sentence_seq = pad_sequences(seq, maxlen=maxlen, truncating="post")

        prediction = model.predict(sentence_seq)[0][0]
        
        # 로그 확인
        print(f"DEBUG: 텍스트='{text}', 예측값={prediction}")
        
        return jsonify({'isToxic': bool(prediction >= 0.5)})
        
    except Exception as e:
        print(f"에러 발생: {e}")
        return jsonify({'isToxic': False})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
