from numpy import array, expand_dims, ndarray
from PIL import Image
from tensorflow.keras import Model
from tensorflow.keras.models import load_model


def init_model(model_name: str) -> Model:
    """
    Load and return a pre-trained Keras model from a specified file.

    This function takes the file path of a saved Keras model (in HDF5 or SavedModel format)
    and loads the model into memory for use. The model can then be used for inference,
    evaluation, or further training.

    Args:
        model_name (str): The path to the saved Keras model file (either .h5 or a SavedModel directory).

    Returns:
        Model: A TensorFlow Keras model instance that can be used for predictions, training, or evaluation.

    Raises:
        OSError: If the file cannot be found or the model cannot be loaded for any reason.
        ValueError: If the model format is invalid or incompatible.
    """
    return load_model(model_name)


def resize_image(file_path: str, target_size: tuple[int, int]) -> ndarray:
    """
    Resizes and preprocesses an image for model input.

    This function takes the file path of an image, resizes it to the specified
    target size, converts the image to RGB format (if it's not already), and
    normalizes the pixel values to a range of [0, 1]. The image is returned as
    a NumPy array with an additional dimension to be compatible with deep
    learning models that expect a batch dimension.

    Args:
        file_path (str): The path to the image file that needs to be resized.
        target_size (tuple[int, int]): The desired (width, height) to resize the image to.

    Returns:
        np.ndarray: A 4D NumPy array of shape (1, height, width, 3) representing the resized,
        normalized image, where the pixel values are scaled to the range [0, 1].

    Steps:
        1. Opens the image from the specified file path.
        2. Converts the image to RGB mode to ensure compatibility with the model.
        3. Resizes the image to the specified target size using high-quality resampling (Lanczos).
        4. Normalizes the pixel values by scaling them from the range [0, 255] to [0, 1].
        5. Expands the array's dimensions by adding a batch dimension at axis 0, as the model
           expects input in the format (batch_size, height, width, channels).

    Notes:
        - The input image is always converted to RGB format, even if the source image is grayscale or RGBA.
        - The resampling method used for resizing is Lanczos, which provides high-quality results
          for shrinking or enlarging images.
    """
    img = Image.open(file_path).convert("RGB")  # Ensure RGB format
    img = img.resize(
        target_size, Image.Resampling.LANCZOS
    )  # Use ANTIALIAS for better quality
    img_array = array(img) / 255.0  # Normalize pixel values
    img_array = expand_dims(img_array, axis=0)
    return img_array
