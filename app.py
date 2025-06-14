from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import os
import tensorflow as tf

app = Flask(__name__)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="smart_pest_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Match these classes exactly with your training folder names
classes = [
    "aphids",
    "armyworm",
    "beetle",
    "bollworm",
    "grasshopper",
    "mites",
    "mosquito",
    "sawfly",
    "stem_borer"
]

solutions = {
    "aphids": "Use neem oil or insecticidal soap.",
    "armyworm": "Apply Bt or introduce natural predators.",
    "beetle": "Use pheromone traps or handpick.",
    "bollworm": "Apply neem or pyrethroid spray.",
    "grasshopper": "Introduce birds or use bait spray.",
    "mites": "Spray miticide or use sulfur-based sprays.",
    "mosquito": "Use larvicide and remove standing water.",
    "sawfly": "Handpick or apply horticultural oil.",
    "stem_borer": "Apply systemic insecticides or trap cropping."
}

def preprocess_image(image):
    image = image.resize((224, 224)).convert("RGB")
    image = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(image, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    img = Image.open(io.BytesIO(request.files['file'].read()))
    input_data = preprocess_image(img)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    class_id = int(np.argmax(output_data))

    if class_id >= len(classes):
        return jsonify({'error': f'Predicted class_id {class_id} is out of range. Model may not match class list.'}), 500

    pest = classes[class_id]
    solution = solutions[pest]

    response = {
        "pest": pest,
        "solution": solution
    }

    print("Sending JSON:", response)
    return jsonify(response), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
