from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import pickle
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

        # [필승 로직] 토크나이저의 속성(attribute)을 쓰지 않고 
        # 직접 텍스트를 숫자로 변환하는 방식(texts_to_sequences)을 강제합니다.
        # 이게 안 되면 tokenizer 객체 자체가 학습된 Keras 토크나이저가 아닙니다.
        
        # 만약 학습 시 사용했던 객체가 .texts_to_sequences 메서드를 가진다면:
        if hasattr(tokenizer, 'texts_to_sequences'):
            seq = tokenizer.texts_to_sequences([text.lower()])
        else:
            # 만약 transformers의 토크나이저라면 .encode 사용
            seq = [tokenizer.encode(text.lower())]

        sentence_seq = pad_sequences(seq, maxlen=maxlen, truncating="post")
        prediction = model.predict(sentence_seq)[0][0]
        
        return jsonify({'isToxic': bool(prediction >= 0.3)})
        
    except Exception as e:
        print(f"디버깅 정보: {e}") # 에러가 나면 Render 로그에 상세히 찍힙니다.
        return jsonify({'isToxic': False}) 

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
