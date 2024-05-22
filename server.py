from flask import Flask, send_file, abort, jsonify, request
import os
import shutil
from flask_cors import CORS
from retrain import Retrain

app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)  # Enable CORS for all routes

@app.route('/')
def index():
    return jsonify(message='Flask server!')

@app.route('/download-model')
def download_file():
    try:
        # Ensure the file exists
        return send_file(f'models/model.tflite', as_attachment=True)
    except FileNotFoundError:
        abort(404)

@app.route('/accept-model', methods=['POST'])
def accept_model():
    # Get the POST data
    data = request.json

    try:
        retrain_model = Retrain(data)

        retrain_model.run()
        return jsonify({"message": data}), 200  
    except FileNotFoundError:
        abort(404)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
