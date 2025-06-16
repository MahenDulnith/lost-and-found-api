from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io
# This is the lightweight TensorFlow Lite runtime
from tflite_runtime.interpreter import Interpreter

# Initialize the Flask app
app = Flask(__name__)

# --- Load the TFLite model and labels ---
try:
    print("Loading TFLite model...")
    interpreter = Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    print("Model loaded successfully!")

    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Loading labels...")
    with open("labels.txt", "r") as f:
        class_labels = [line.strip() for line in f.readlines()]
    print("Labels loaded successfully:", class_labels)

except Exception as e:
    print(f"Error loading model or labels: {e}")
    interpreter = None
    class_labels = None

@app.route("/")
def index():
    return "Lightweight TFLite Model Server is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if not interpreter or not class_labels:
        return jsonify({"error": "Model or labels not loaded"}), 503

    if 'image' not in request.files:
        return jsonify({"error": "No image file in request"}), 400

    file = request.files['image']

    try:
        image = Image.open(file.stream).convert('RGB')

        # Preprocess the image to match the model's input requirements
        # Get the required input size from the model's input details
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]
        image = image.resize((width, height))

        # Convert image to numpy array and normalize
        input_data = np.expand_dims(image, axis=0)

        # TFLite models exported from Teachable Machine (floating point)
        # often expect input values normalized to the [-1, 1] range.
        input_data = (np.float32(input_data) - 127.5) / 127.5

        # Set the tensor, invoke the interpreter, and get the result
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        prediction_scores = interpreter.get_tensor(output_details[0]['index'])[0]

        # Format the response into the format Bubble expects
        predictions = []
        for i, score in enumerate(prediction_scores):
            # We need to strip the number prefix from the label, e.g., "0 Keys" -> "Keys"
            label_text = class_labels[i].split(' ', 1)[1]
            predictions.append({
                "className": label_text,
                "probability": float(score)
            })

        predictions.sort(key=lambda x: x['probability'], reverse=True)
        print("Top prediction:", predictions[0])

        return jsonify(predictions)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": "Failed to predict", "details": str(e)}), 500

# This runs the app
app.run(host='0.0.0.0', port=8080)
