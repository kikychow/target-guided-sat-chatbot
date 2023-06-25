from flask import Flask, request, jsonify
from flask_cors import CORS
# from model_old import get_response
from model import get_response

app = Flask(__name__)
CORS(app)


@app.route('/predict', methods=['POST'])
def predict():
    text = request.get_json().get("message")
    response = get_response(text)
    message = {"success": True, "answer": response}
    return jsonify(message)


@app.route('/hello')
def hello():
    return 'Hello, World'


if __name__ == "__main__":
    app.run(debug=True)
