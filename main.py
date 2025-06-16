from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io
from tflite_runtime.interpreter import Interpreter

# Initialize the Flask app
app = Flask(__name__)

# --- Load the TFLite model and labels ---
try:
    print("Loading TFLite model...")
    interpreter = Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    print("Model loaded successfully!")

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Loading labels...")
    with open("labels.txt", "r") as f:
        # Clean up labels by removing number prefix and extra spaces
        class_labels = [line.strip().split(' ', 1)[1] for line in f.readlines()]
    print("Labels loaded successfully:", class_labels)

except Exception as e:
    print(f"Error loading model or labels: {e}")
    interpreter = None
    class_labels = None

@app.route("/")
def index():
    # A simple route to check if the server is running
    return "Python TFLite Model Server is running on Render!"

@app.route("/predict", methods=["POST"])
def predict():
    if not interpreter or not class_labels:
        return jsonify({"error": "Model or labels not loaded"}), 503

    if 'image' not in request.files:
        return jsonify({"error": "No image file in request"}), 400

    file = request.files['image']
    
    try:
        image = Image.open(file.stream).convert('RGB')
        
        # Preprocess the image
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]
        image = image.resize((width, height))
        input_data = np.expand_dims(image, axis=0)
        input_data = (np.float32(input_data) - 127.5) / 127.5

        # Run prediction
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        prediction_scores = interpreter.get_tensor(output_details[0]['index'])[0]

        # Format the response
        predictions = []
        for i, score in enumerate(prediction_scores):
            predictions.append({
                "className": class_labels[i],
                "probability": float(score)
            })
        
        predictions.sort(key=lambda x: x['probability'], reverse=True)
        print("Top prediction:", predictions[0])

        return jsonify(predictions)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": "Failed to predict", "details": str(e)}), 500

# This part is needed for some environments but gunicorn handles it on Render
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
