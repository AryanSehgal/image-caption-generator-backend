import sys
import os
from io import BytesIO

import cloudinary as Cloud
import cloudinary.uploader as cu
from requests import get as rq
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

import caption_gen

app = Flask(__name__)
CORS(app)  # allows a separate frontend (on a different domain) to call this API

Cloud.config.update(
    cloud_name=os.environ.get('CLOUD_NAME'),
    api_key=os.environ.get('API_KEY'),
    api_secret=os.environ.get('API_SECRET')
)

@app.route('/')
def hello():
    return jsonify({"status": "Image captioning API is running"})

@app.route('/caption', methods=['POST'])
def caption_image():
    if 'userfile' not in request.files:
        return jsonify({"error": "No file uploaded. Send it as 'userfile' in form-data."}), 400

    file_to_upload = request.files['userfile']

    try:
        upload_result = cu.upload(file_to_upload)
        response = rq(upload_result["url"])
        img = Image.open(BytesIO(response.content))
        caption = caption_gen.caption_this_image(img)

        return jsonify({
            "image": upload_result["url"],
            "caption": caption
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run()