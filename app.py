from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app) # 웹 앱(GitHub Pages)에서 서버로 데이터를 보낼 수 있게 허용

@app.route('/check', methods=['POST'])
def check():
    text = request.json.get('text', '')
    # 여기에 korcen-kogpt2의 추론 로직(predict_text 함수)을 가져와 붙이세요.
    # 결과값 0.5 이상이면 True, 아니면 False 반환
    return jsonify({'isToxic': True}) 

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
