from flask import Flask, request
from core_predict import CorePredict
import numpy as np
import cv2

app = Flask(__name__)

@app.get("/")
def check():
    return "OK!"

@app.post("/predict")
def predict_height():

    image_file = request.files['image'].read()

    age = request.form.get('data')

    nparr = np.fromstring(image_file, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return CorePredict().predict(image, int(age))

if __name__ == '__main__':
    app.run(debug=True)