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
        data = request.json
        text = str(data.get('text', '')).lower()
        
        # 1. 전처리 로직 (깃허브 main_kogpt2.py 로직)
        encoded = tokenizer.encode_plus(text, max_length=1000, padding="max_length", truncation=True)['input_ids']
        sentence_seq = pad_sequences([encoded], maxlen=1000, truncating="post", padding="post")
        
        # 2. 모델 예측
        prediction = model.predict(sentence_seq)
        
        # 3. 점수 확인 (로그 출력)
        score = float(prediction[0][0])
        print(f"DEBUG: 문장='{text}', 예측값={score}")
        
        is_toxic = bool(score > 0.9)
        return jsonify({'isToxic': is_toxic})
        
    except Exception as e:
        # [중요] 에러가 나면 정확히 어떤 에러인지 서버 로그에 남김
        import traceback
        traceback.print_exc() 
        return jsonify({'isToxic': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
