from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import pickle
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app)

# 1. 모델과 토크나이저 로드
try:
    model = tf.keras.models.load_model('vdcnn_model_with_kogpt2.h5')
    with open("tokenizer_with_kogpt2.pickle", "rb") as f:
        tokenizer = pickle.load(f)
    model_loaded = True
except Exception as e:
    print(f"로드 오류: {e}")
    model_loaded = False

@app.route('/healthz', methods=['GET'])
def healthz():
    return "OK", 200

@app.route('/check', methods=['POST'])
def check_toxic():
    if not model_loaded: return jsonify({'isToxic': False})
    
    try:
        data = request.json
        text = str(data.get('text', '')).lower()
        if not text: return jsonify({'isToxic': False})
        
        # [수정된 핵심 로직] 
        # 선생님 예시 코드의 encode_plus만 사용합니다.
        # texts_to_sequences를 완전히 제거했습니다.
        encoded_sentence = tokenizer.encode_plus(
            text,
            max_length=1000,
            padding="max_length",
            truncation=True
        )['input_ids']
        
        sentence_seq = pad_sequences([encoded_sentence], maxlen=1000, truncating="post")
        prediction = model.predict(sentence_seq)[0][0]
        
        print(f"DEBUG_SCORE: 텍스트='{text}', 예측점수={prediction}")
        
        # 0.5 이상이면 True 반환
        return jsonify({'isToxic': bool(prediction >= 0.5)})
        
    except Exception as e:
        # 에러가 나면 굳이 False를 뱉지 말고, 
        # 에러 내용을 로그에 찍어서 확인해야 합니다.
        print(f"상세 에러 내용: {e}")
        return jsonify({'isToxic': False})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
