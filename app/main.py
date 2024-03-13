from flask import Flask, render_template, request, jsonify
from script import *

app = Flask(__name__, template_folder='templates')

@ app.route('/', methods=['POST', 'GET'])
def main():
    if request.method == 'GET':
        return render_template('index.html')

@ app.route('/read', methods=['POST', 'GET'])
def model_inference():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        inputData = request.json
        sentence1 = inputData['input_a'] 
        sentence2 = inputData['input_b']

        result = inference(sentence1, sentence2)
        result = {'student inference': result}

        return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

