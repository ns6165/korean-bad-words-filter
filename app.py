from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app)

# 모델과 토크나이저 로드
model = tf.keras.models.load_model('vdcnn_model_with_kogpt2.h5')
with open("tokenizer_with_kogpt2.pickle", "rb") as f:
    tokenizer = pickle.load(f)

maxlen = 1000

@app.route('/check', methods=['POST'])
def check_toxic():
    data = request.json
    text = data.get('text', '')
    
    # [수정된 부분] 
    # encode_plus 대신, 선생님이 학습 때 쓰셨던 방식처럼 
    # tokenizer 객체 내부에 존재하는 encode 함수를 명시적으로 찾거나,
    # 만약 계속 안 된다면 tokenizer.encode()를 직접 호출해 보세요.
    try:
        encoded_sentence = tokenizer.encode(text.lower(), max_length=maxlen, padding="max_length", truncation=True)
    except:
        # 혹시 encode가 안 될 경우 대비 (학습시 사용된 구조에 따라 다름)
        encoded_sentence = tokenizer.encode(text.lower())
    
    sentence_seq = pad_sequences([encoded_sentence], maxlen=maxlen, truncating="post")
    prediction = model.predict(sentence_seq)[0][0]
    
    return jsonify({'isToxic': bool(prediction >= 0.5)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
