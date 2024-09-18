from os.path import join

import numpy as np
from flask import Flask, jsonify, redirect, render_template, request, url_for
from PIL import Image
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from werkzeug.datastructures import FileStorage
from werkzeug.wrappers import Response

app: Flask = Flask(__name__)
TARGET_SIZE: tuple[int] = (512, 512)
UPLOAD_FOLDER: str = "static"
CATEGORIES: tuple[str] = ("glioma", "meningioma", "pituitary_tumor")


def init_model(model_name: str) -> Model:
    return load_model(model_name)


def resize_image(file_path: str, target_size: tuple[int]):
    img = Image.open(file_path).convert("RGB")  # Ensure RGB format
    img = img.resize(target_size, Image.LANCZOS)  # Use ANTIALIAS for better quality
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# Routes
@app.route("/")
def home() -> str:
    """Render the home page"""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict() -> Response:
    """Handle image upload and prediction."""
    try:
        file: FileStorage = request.files["file"]
        if file is None or file.filename == "":
            raise ValueError("No file is selected")
        file_path = join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        image = resize_image(file_path, TARGET_SIZE)
        prediction = init_model("brain_tumor_classifier.keras").predict(image)
        result: str = CATEGORIES[np.argmax(prediction)]
        confidence_rates = np.fromstring(str(prediction).strip("[]"), sep=" ")
        print(f"Prediction: {prediction}\nConfidence rates: {confidence_rates}")

        return redirect(url_for("results", filename=file.filename, result=result))
    except KeyError:
        data = jsonify({"error": "No file part"})
        data.status_code = 400
        return data
    except Exception as e:
        app.logger.error(f"Error processing request: {e}")
        data = jsonify({"error": "No file part"})
        data.status_code = 500
        return data


@app.route("/results/<filename>")
def results(filename) -> str:
    """Render the resulsts page with the prediction data"""
    result = request.args.get("result")
    return render_template("results.html", filename=filename, result=result)


if __name__ == "__main__":
    app.run(debug=True)
