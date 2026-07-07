from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import pickle
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app)

# 모델 및 토크나이저 로드
model = tf.keras.models.load_model('vdcnn_model_with_kogpt2.h5')
with open("tokenizer_with_kogpt2.pickle", "rb") as f:
    tokenizer = pickle.load(f)

maxlen = 1000

@app.route('/check', methods=['POST'])
def check_toxic():
    try:
        text = request.json.get('text', '')
        
        # 1. 텍스트 변환: KoGPT2 토크나이저는 대개 .encode() 메서드 하나만 가집니다.
        # .encode_plus나 .texts_to_sequences를 버리고 가장 단순한 .encode()만 사용합니다.
        # 만약 이것도 안 되면 tokenizer 객체의 구조가 매우 특이한 것이니 
        # 그냥 단순히 리스트로 변환하는 방식을 씁니다.
        try:
            tokens = tokenizer.encode(text) 
        except:
            # 위가 안 되면 tokenizer 객체를 출력해서 확인해야 하므로 예외 처리
            tokens = tokenizer.encode_plus(text)['input_ids']
            
        sentence_seq = pad_sequences([tokens], maxlen=maxlen, truncating="post")
        
        # 2. 예측
        prediction = model.predict(sentence_seq)[0][0]
        
        # 3. 비속어 판별 (임계값 0.3으로 조정)
        is_toxic = bool(prediction >= 0.3)
        
        return jsonify({'isToxic': is_toxic})
        
    except Exception as e:
        print(f"DEBUG: {e}")
        return jsonify({'isToxic': False})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
