from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

app = Flask(__name__)
CORS(app)

# 모델 로드
model = tf.keras.models.load_model('vdcnn_model_with_kogpt2.h5')
# 토크나이저 로드
with open("tokenizer_with_kogpt2.pickle", "rb") as f:
    tokenizer = pickle.load(f)

maxlen = 1000

@app.route('/check', methods=['POST'])
def check_toxic():
    try:
        data = request.json
        text = data.get('text', '')
        if not text: return jsonify({'isToxic': False})

        # --- 수정 핵심: Tokenizer 라이브러리 호출을 최소화 ---
        # tokenize 방식이 버전별로 충돌나므로, 
        # 단순히 텍스트를 시퀀스로 변환하는 가장 기초적인 함수를 사용
        
        # 만약 아래 코드가 또 에러가 나면, tokenizer 객체 내용을 출력해서 확인해야 합니다.
        seq = tokenizer.texts_to_sequences([text.lower()]) 
        sentence_seq = pad_sequences(seq, maxlen=maxlen, truncating="post")
        
        prediction = model.predict(sentence_seq)[0][0]
        
        # 임계값을 0.3 정도로 낮추어 민감도를 높입니다. (욕설이 아니라고 판정될 때)
        return jsonify({'isToxic': bool(prediction >= 0.3)})
        
    except Exception as e:
        print(f"상세 에러 내용: {e}")
        return jsonify({'isToxic': False}) # 에러 시 등록 허용

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
