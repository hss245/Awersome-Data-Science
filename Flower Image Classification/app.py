from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from inference import ImageClassificationInference

app = Flask(__name__)
UPLOAD_FOLDER = "localStorage/"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = tf.keras.models.load_model("flowerTFModel")

@app.route('/', methods = ['GET'])
def homepage():
    return render_template('index.html')

@app.route('/serveractive', methods = ['GET'])
def serveractive():
    return {'Success' : 'Server is Active. Congrats'}

@app.route('/prediction', methods = ['POST'])
def prediction():
    if request.method == "POST":
        fileName = request.files["fileUpload"]
        _fileName = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(fileName.filename))
        fileName.save(_fileName)
        imgInference = ImageClassificationInference()
        prediction = imgInference.modelInferencing(_fileName, model)
        os.remove(_fileName)
        return render_template("index.html", returnedText = f"Prediction is {prediction}")

if __name__ == "__main__":
    app.run(host = "0.0.0.0",
            port = 5001, 
            debug = False)