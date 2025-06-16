import functions_framework
from flask import jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

# --- Load the model and labels (globally) ---
# This runs only once when the function instance starts, making it fast.
try:
    print("Loading Keras model...")
    model = load_model("model.tflite") # Note: Keras can load .h5 and .tflite
    print("Model loaded successfully!")

    print("Loading labels...")
    with open("labels.txt", "r") as f:
        # Create a clean list of labels without the numbers
        class_labels = [line.strip().split(' ', 1)[1] for line in f]
    print("Labels loaded successfully:", class_labels)

    # Get input shape for resizing
    input_shape = model.get_input_details()[0]['shape']
    IMG_HEIGHT = input_shape[1]
    IMG_WIDTH = input_shape[2]

except Exception as e:
    print(f"Error loading model or labels: {e}")
    model = None
    class_labels = None

# This decorator turns the function into an HTTP-triggered Cloud Function
@functions_framework.http
def predict(request):
    # --- CORS Headers (for allowing requests from Bubble) ---
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Max-Age': '3600'
    }
    if request.method == 'OPTIONS':
        return ('', 204, headers)

    if not model or not class_labels:
        return (jsonify({"error": "Model or labels not loaded"}), 503, headers)

    if 'image' not in request.files:
        return (jsonify({"error": "No image file in request"}), 400, headers)

    file = request.files['image']

    try:
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        image = image.resize((IMG_WIDTH, IMG_HEIGHT))
        image_array = np.asarray(image)

        # Normalize the image to the [-1, 1] range
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Create the input tensor
        data = np.expand_dims(normalized_image_array, axis=0)

        # Make a prediction
        prediction_scores = model.predict(data)[0]

        # Format the response
        predictions = []
        for i, score in enumerate(prediction_scores):
            predictions.append({
                "className": class_labels[i],
                "probability": float(score)
            })

        predictions.sort(key=lambda x: x['probability'], reverse=True)
        print("Top prediction:", predictions[0])

        return (jsonify(predictions), 200, headers)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return (jsonify({"error": "Failed to predict", "details": str(e)}), 500, headers)
