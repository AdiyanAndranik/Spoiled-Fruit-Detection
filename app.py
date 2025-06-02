import sys
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from spoiledDetection.pipeline.training_pipeline import TrainPipeline
from spoiledDetection.utils.main_utils import decodeImage, encodeImageIntoBase64
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS, cross_origin
from spoiledDetection.constants.application import APP_HOST, APP_PORT

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.model = tf.keras.models.load_model("artifacts/model_trainer/cnn_model.h5")
        self.class_names = ['Թարմ', 'Փչացած']

@app.route("/train")
def trainRoute():
    obj = TrainPipeline()
    obj.run_pipeline()
    return "Training Successful!!"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST', 'GET'])
@cross_origin()
def predictRoute():
    try:
        image = request.json['image']
        decodeImage(image, clApp.filename)

        img = Image.open(clApp.filename).convert("RGB")
        img = img.resize((64, 64))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = clApp.model.predict(img_array)
        predicted_class = clApp.class_names[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        img = Image.open(clApp.filename)
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        # text = f"{predicted_class} ({confidence:.2f})"
        # draw.text((10, 10), text, fill="red", font=font)
        img.save(clApp.filename)

        opencodedbase64 = encodeImageIntoBase64(clApp.filename)
        result = {
            "image": opencodedbase64.decode('utf-8'),
            "prediction": predicted_class,
            "confidence": confidence
        }

    except ValueError as val:
        print(val)
        return Response("Value not found inside json data")
    except KeyError:
        return Response("Key value error incorrect key passed")
    except Exception as e:
        print(e)
        return Response("Invalid input")

    return jsonify(result)

if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host=APP_HOST, port=APP_PORT)