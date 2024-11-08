from os.path import join

from flask import Flask, jsonify, redirect, render_template, request, url_for
from numpy import argmax, fromstring, ndarray
from werkzeug.datastructures import FileStorage
from werkzeug.wrappers import Response

from utils import init_model, resize_image

APP: Flask = Flask(__name__)
TARGET_SIZE: tuple[int, int] = (512, 512)  # Target size for resizing the images
UPLOAD_FOLDER: str = "static/upload"  # Brain scans which users upload go here
CATEGORIES: tuple[str, str, str] = (
    "glioma",
    "meningioma",
    "pituitary_tumor",
)  # Categories of the tumors


# Routes
@APP.route("/")
def home() -> str:
    """
    Renders the index.html template, which is the home page of the application.

    Returns:
        str: The rendered HTML response for the home page.
    """
    return render_template("index.html")


@APP.route("/upload", methods=["GET"])
def upload() -> Response:
    """
    Renders the upload.html template, which is the page where users can upload brain scan images.

    Returns:
        Response: The rendered HTML response for the upload page.
    """
    return render_template("upload.html")


@APP.route("/predict", methods=["POST"])
def predict() -> Response:
    """
    Handle image upload, resize the image, perform a prediction using a pre-trained
    brain tumor classification model, and redirect to the results page with prediction data.

    Workflow:
    1. The function receives an uploaded image file through an HTTP POST request.
    2. It checks if the uploaded file exists and is valid.
    3. The image is saved to the `UPLOAD_FOLDER` directory.
    4. The image is resized to match the input size required by the model (512x512 pixels).
    5. The pre-trained model (`brain_tumor_classifier.keras`) is loaded, and the resized image is fed into the model.
    6. The model outputs a prediction representing the type of brain tumor (glioma, meningioma, or pituitary tumor).
    7. The function calculates the confidence levels for each tumor type.
    8. The prediction result (tumor type) and the confidence rates are passed as arguments to the results page, where they are displayed.

    Returns:
        Response: Redirects to the "/results" page with the filename, predicted tumor type,
        and the confidence levels for each class.

    Raises:
        ValueError: If no file is uploaded or an invalid file is provided.
        KeyError: If the request does not contain a file in the expected key "file".
        Exception: For any other errors during image processing or prediction.

    Parameters:
    - POST request containing a file uploaded through an HTML form.

    Exceptions Handled:
    - A KeyError is raised if the file part is missing in the request.
    - If no file is selected or an invalid file is uploaded, a ValueError is raised.
    - Any other exception, such as issues during image processing, saving, or prediction,
      is logged, and a 500 Internal Server Error response is returned.

    Redirects:
    - Upon successful prediction, the function redirects to the `/results` route, passing:
        - filename: The original filename of the uploaded image.
        - result: The predicted tumor type (e.g., "Glioma", "Meningioma", "Pituitary Tumor").
        - conf1, conf2, conf3: Confidence rates for the three categories in percentage format.
    """
    try:
        file: FileStorage | None = request.files.get("file")
        if not file:
            raise ValueError("No file is selected")
        elif isinstance(file.filename, str):
            file_path = join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

        image = resize_image(file_path, TARGET_SIZE)
        prediction = init_model("brain_tumor_classifier.keras").predict(image)
        result: str = CATEGORIES[argmax(prediction)].replace("_", " ").title()
        confidence_rates: ndarray = fromstring(str(prediction).strip("[]"), sep=" ")

        return redirect(
            url_for(
                "results",
                filename=file.filename,
                result=result,
                conf1=f"{confidence_rates[0] * 100.0:.3f}",
                conf2=f"{confidence_rates[1] * 100.0:.3f}",
                conf3=f"{confidence_rates[2] * 100.0:.3f}",
            )
        )
    except KeyError:
        data = jsonify({"error": "No file part"})
        data.status_code = 400
        return data
    except Exception as e:
        APP.logger.error(f"Error processing request: {e}")
        data = jsonify({"error": "No file part"})
        data.status_code = 500
        return data


@APP.route("/results/<filename>")
def results(filename: str) -> str:
    """
    Render the results page with the prediction data and confidence rates.

    This function handles the route for displaying the results of the brain tumor
    classification prediction after the user uploads an image. It expects the following
    query parameters to be present in the request:

    - result: A string representing the predicted class of the brain tumor
      (e.g., 'glioma', 'meningioma', or 'pituitary_tumor').
    - conf1: A float representing the confidence score for the first category (glioma).
    - conf2: A float representing the confidence score for the second category (meningioma).
    - conf3: A float representing the confidence score for the third category (pituitary tumor).

    These confidence scores represent how likely the model believes the image belongs
    to each of the three categories. The filename of the uploaded image is passed as a
    URL parameter and displayed on the results page along with the prediction data.

    Args:
        filename (str): The name of the uploaded image file, passed as a URL parameter.

    Returns:
        str: The rendered HTML content for the results page, which includes the
        predicted class and the confidence scores for each tumor type.
    """
    result = request.args.get("result")
    return render_template(
        "results.html",
        filename=f"upload/{filename}",
        result=result,
        conf1=request.args.get("conf1"),
        conf2=request.args.get("conf2"),
        conf3=request.args.get("conf3"),
    )


if __name__ == "__main__":
    APP.run(debug=True)
