from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import pickle
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app)

# 1. 모델과 토크나이저 로드 (학습 때 사용한 모델과 사전)
model = tf.keras.models.load_model('vdcnn_model_with_kogpt2.h5')
with open("tokenizer_with_kogpt2.pickle", "rb") as f:
    tokenizer = pickle.load(f)

# 2. 토크나이저 내의 사전(vocab) 직접 추출
# 이렇게 하면 라이브러리 메서드 호출 없이 사전만 사용하므로 버전 에러가 안 납니다.
vocab = getattr(tokenizer, 'vocab', {})
if not vocab and hasattr(tokenizer, 'word_index'):
    vocab = tokenizer.word_index

@app.route('/healthz', methods=['GET'])
def healthz():
    return "OK", 200

@app.route('/check', methods=['POST'])
def check_toxic():
    try:
        data = request.json
        text = str(data.get('text', '')).lower()
        
        # 3. 사전 기반 토크나이징 (가장 안전한 방식)
        # 단어를 쪼개고, 사전에 있으면 숫자로, 없으면 0으로 바꿈
        tokens = text.split()
        seq = [[vocab.get(w, 0) for w in tokens]]
        
        # 4. 모델 입력 맞추기 (maxlen 1000)
        sentence_seq = pad_sequences(seq, maxlen=1000, truncating="post", padding="post")
        
        # 5. 모델 예측
        prediction = model.predict(sentence_seq)[0][0]
        
        # 확인용 로그
        print(f"DEBUG_SCORE: 텍스트='{text}', 예측값={prediction}")
        
        # 0.5 이상이면 욕설로 판정
        return jsonify({'isToxic': bool(prediction >= 0.5)})
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        return jsonify({'isToxic': False})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
