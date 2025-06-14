from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import os
import tensorflow as tf

app = Flask(__name__)

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="smart_pest_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Update this list to match your model's output classes
classes = ["Aphid", "Whitefly", "Caterpillar", "Thrips", "Grasshopper"]

# Pest control solutions
solutions = {
    "Aphid": "Use neem oil",
    "Whitefly": "Use yellow sticky traps",
    "Caterpillar": "Apply Bt spray",
    "Thrips": "Use reflective mulch",
    "Grasshopper": "Introduce birds"
}

# Preprocess image before passing to the model
def preprocess_image(image):
    image = image.resize((224, 224)).convert("RGB")
    image = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(image, axis=0)

# API route to handle pest prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    try:
        img = Image.open(io.BytesIO(request.files['file'].read()))
        input_data = preprocess_image(img)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        class_id = int(np.argmax(output_data))

        print("Predicted class ID:", class_id)
        print("Model raw output:", output_data)

        if class_id >= len(classes):
            return jsonify({"error": f"Predicted class_id {class_id} is out of range. Model may not match class list."}), 500

        pest = classes[class_id]
        response = {
            "pest": pest,
            "solution": solutions.get(pest, "No solution available")
        }

        print("Sending JSON:", response)
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": f"Exception occurred: {str(e)}"}), 500

# Run the Flask app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
