from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer # 코드가 KoGPT2 기반이므로 필요

app = Flask(__name__)
CORS(app) # 웹 앱에서 서버로 접근 가능하게 설정

# 1. 모델과 토크나이저 로드
model = tf.keras.models.load_model('vdcnn_model_with_kogpt2.h5')
# 토크나이저 로드 (tokenizer_with_kogpt2.pickle 사용)
with open("tokenizer_with_kogpt2.pickle", "rb") as f:
    tokenizer = pickle.load(f)

maxlen = 1000 

@app.route('/check', methods=['POST'])
def check_toxic():
    data = request.json
    text = data.get('text', '')
    
    # 2. 판별 로직 (main_kogpt2.py의 predict_text 참고)
    sentence = text.lower()
    # 주의: 토크나이저 사용법은 저장소의 구현 방식에 맞게 조정이 필요할 수 있습니다
    encoded_sentence = tokenizer.encode_plus(
        sentence,
        max_length=maxlen,
        padding="max_length",
        truncation=True
    )['input_ids']
    
    sentence_seq = pad_sequences([encoded_sentence], maxlen=maxlen, truncating="post")
    prediction = model.predict(sentence_seq)[0][0]
    
    # 0.5 이상이면 욕설로 판별
    is_toxic = bool(prediction >= 0.5)
    
    return jsonify({'isToxic': is_toxic})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
