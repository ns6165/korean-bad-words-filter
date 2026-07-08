from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import pickle
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app)

# 1. 모델과 토크나이저 로드 (학습 시와 동일한 환경/파일)
model = tf.keras.models.load_model('vdcnn_model_with_kogpt2.h5')
with open("tokenizer_with_kogpt2.pickle", "rb") as f:
    tokenizer = pickle.load(f)
maxlen = 1000

@app.route('/healthz', methods=['GET'])
def health():
    return '', 200

@app.route('/check', methods=['POST'])
def check_toxic():
    try:
        text = str(request.json.get('text', '')).lower()
        
        # 2. 깃허브(main_kogpt2.py)의 로직 그대로 사용
        encoded = tokenizer.encode_plus(text, max_length=maxlen, padding="max_length", truncation=True)['input_ids']
        sentence_seq = pad_sequences([encoded], maxlen=maxlen, truncating="post", padding="post")
        
        # 3. 모델 예측
        prediction = model.predict(sentence_seq)[0][0]
        is_toxic = bool(prediction > 0.9)
        
        return jsonify({'isToxic': is_toxic})
    except Exception as e:
        return jsonify({'isToxic': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
