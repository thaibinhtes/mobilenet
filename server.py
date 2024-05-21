from flask import Flask, send_file, abort, jsonify
import os
import shutil
from flask_cors import CORS

app = Flask(__name__)
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

@app.route('/accpet-model')
def accept_model():
  filename = 'model.tflite'
  src = os.path.join('models', filename)
  dest = os.path.join('tested', filename)
  
  # Check if the source file exists
  if not os.path.exists(src):
      abort(404, description="Source file not found.")

  try:
      shutil.move(src, dest)
      return jsonify({"message": f"File '{filename}' moved successfully."}), 200
  except Exception as e:
      abort(500, description=str(e))

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
