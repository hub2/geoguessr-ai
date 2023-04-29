import os
import hashlib
import json
from flask import Flask, flash, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import base64
from flask_cors import CORS, cross_origin
from eval_one import eval_one, load_model, eval_multi


app = Flask(__name__)
CORS(app)

model = load_model("2023-04-29-geoguessr-15.pth")
@app.route("/eval", methods=["GET", "POST"])
@cross_origin()
def upload_file():
    if request.method == "POST":
        r = request.data
        parsed = json.loads(r)
        pano = parsed['pano']
        lat = parsed['lat']
        lon = parsed['lon']

        with open("panos.txt", "a") as f:
            f.write(f"{lat},{lon}")
            f.write("\n")

        class_, (lat, lng) = eval_one(model, panoid=pano)
        #class_, (lat, lng) = eval_multi(model, coords=(lat, lon))
        print(class_, lat, lng)
        #lng, lat = lat, lng
        response = jsonify({"lat": float(lat), "lng": float(lng)})
        #response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    return ""


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8000)

