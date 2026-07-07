from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import pickle
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app)

# KoGPT2 모델 구조에 맞는 로드
maxlen = 1000
model = tf.keras.models.load_model('vdcnn_model_with_kogpt2.h5')
with open("tokenizer_with_kogpt2.pickle", "rb") as f:
    tokenizer = pickle.load(f)

@app.route('/check', methods=['POST'])
def check_toxic():
    try:
        data = request.json
        text = data.get('text', '')
        
        # main_kogpt2.py의 핵심 로직 (encode_plus)
        encoded_sentence = tokenizer.encode_plus(
            text.lower(),
            max_length=maxlen,
            padding="max_length",
            truncation=True
        )['input_ids']

        sentence_seq = pad_sequences([encoded_sentence], maxlen=maxlen, truncating="post")
        prediction = model.predict(sentence_seq)[0][0]
        
        # 모델 예측값 로그 출력 (디버깅용)
        print(f"DEBUG: Input='{text}', Score={prediction}")
        
        return jsonify({'isToxic': bool(prediction >= 0.5)})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'isToxic': False})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
