from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import os

# Load trained model
model_path = r"C:\\Users\\kansa\\OneDrive\\Desktop\\Image classification\\cifar10_flask_app\\model\\cifar10_model.h5"
if os.path.exists(model_path):
    model = load_model(model_path)
    print("✅ Model loaded successfully.")
else:
    raise FileNotFoundError(f"❌ File not found: {model_path}")

# Define class labels
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Initialize Flask app
app = Flask(__name__)

# Image preprocessing function
def preprocess_image(image):
    img = Image.open(image).convert('RGB')
    img = img.resize((32, 32))
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Home route (index.html page)
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'})

    try:
        processed_image = preprocess_image(file)
        prediction = model.predict(processed_image)
        predicted_label = CLASS_NAMES[np.argmax(prediction)]
        confidence = float(np.max(prediction)) * 100
        result_text = f"Prediction: {predicted_label} ({confidence:.2f}%)"
        return render_template("index.html", prediction=result_text)
    except Exception as e:
        return jsonify({'error': str(e)})

# Run app
if __name__ == '__main__':
    app.run(debug=True)
