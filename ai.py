import os
import hashlib
import json
from flask import Flask, flash, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import base64
from flask_cors import CORS, cross_origin
from eval_one import eval_one


UPLOAD_FOLDER = './ai_data'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)

@app.route("/eval", methods=["GET", "POST"])
@cross_origin()
def upload_file():
    if request.method == "POST":
        r = request.data
        parsed = json.loads(r)
        # print(parsed)
        folder = app.config["UPLOAD_FOLDER"]
        os.makedirs(folder, exist_ok=True)
        img = base64.b64decode(parsed["img"].split(",")[1])
        path = os.path.join(folder, "test.png")
        with open(path, "wb") as f:
            f.write(img)
        lat, lng = eval_one(path)
        print(lat, lng)
        response = jsonify({"lat": float(lat), "lng": float(lng)})
        #response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    return ""


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8000)

