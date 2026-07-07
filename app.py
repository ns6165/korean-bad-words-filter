from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import pickle
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app)

# 1. 모델과 토크나이저 로드 (예시 코드와 동일한 경로)
model_path = 'vdcnn_model_with_kogpt2.h5'
tokenizer_path = "tokenizer_with_kogpt2.pickle"

# 모델 로드
model = tf.keras.models.load_model(model_path)
# 토크나이저 로드
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)

@app.route('/check', methods=['POST'])
def check_toxic():
    try:
        data = request.json
        text = data.get('text', '')
        
        # 2. 예시 코드의 텍스트 전처리 방식 그대로 사용
        sentence = text.lower()
        
        # 3. [핵심] 예시 코드의 encode_plus 로직 그대로 적용
        # tokenizer 객체의 encode_plus를 사용하여 모델 입력 생성
        encoded_sentence = tokenizer.encode_plus(
            sentence,
            max_length=1000,
            padding="max_length",
            truncation=True
        )['input_ids']
        
        # 모델 입력 형상 맞추기
        sentence_seq = pad_sequences([encoded_sentence], maxlen=1000, truncating="post")
        
        # 4. 모델 예측
        prediction = model.predict(sentence_seq)[0][0]
        
        # 로그 확인용: 점수가 어떻게 나오는지 서버 로그에 찍어줍니다.
        print(f"DEBUG: Input='{text}', Score={prediction}")
        
        # 0.5 이상이면 True로 판정
        return jsonify({'isToxic': bool(prediction >= 0.5)})
        
    except Exception as e:
        # 에러 발생 시 로그 확인 (여기에 에러 내용이 찍힐 겁니다)
        print(f"ERROR: {str(e)}")
        return jsonify({'isToxic': False})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
