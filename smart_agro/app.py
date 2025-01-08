from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import pickle
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

# Configuration for file uploads
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load the models
CROP_MODEL_PATH = 'static/models/crop_prediction_model.pkl'
LABEL_ENCODER_PATH = 'static/models/label_encoder.pkl'
DISEASE_MODEL_PATH = 'static/models/disease_detection_model.h5'

# Load models
with open(CROP_MODEL_PATH, 'rb') as f:
    crop_model = pickle.load(f)

with open(LABEL_ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)
    
disease_model = load_model(DISEASE_MODEL_PATH)

# function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# function to preprocess and predict disease
def predict_disease(img_path):
    img = Image.open(img_path)
    img = img.resize((224, 224))  # Resize to model's expected input size
    img = np.array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = disease_model.predict(img)
    classes = ['imagehealthy', 'potatoearlyblight', 'potatolateblight']  # Class labels
    return classes[np.argmax(prediction)]

@app.route('/')
def home():
    return render_template('home.html', title="Home")

@app.route('/crop-recommendation', methods=['GET', 'POST'])
def crop_recommendation():
    if request.method == 'POST':
        N = float(request.form.get('N'))
        P = float(request.form.get('P'))
        K = float(request.form.get('K'))
        temperature = float(request.form.get('temperature'))
        humidity = float(request.form.get('humidity'))
        ph = float(request.form.get('ph'))
        rainfall = float(request.form.get('rainfall'))

        # Prepare input data for the model
        input_data = [[N, P, K, temperature, humidity, ph, rainfall]]
        prediction = crop_model.predict(input_data)[0]  
        crop_name = label_encoder.inverse_transform([prediction])[0]  

        return render_template('crop_recommendation.html', title="Crop Recommendation", prediction=crop_name)

    return render_template('crop_recommendation.html', title="Crop Recommendation")

@app.route('/disease-detection', methods=['GET', 'POST'])
def disease_detection():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('disease_detection.html', title="Disease Detection", error="No file part")
        file = request.files['file']
        if file.filename == '':
            return render_template('disease_detection.html', title="Disease Detection", error="No selected file")

        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Predict disease using the uploaded image
            result = predict_disease(filepath)

            return render_template('disease_detection.html', title="Disease Detection", result=result)

    return render_template('disease_detection.html', title="Disease Detection")

@app.route('/details')
def details():
    return render_template('details.html', title="Details")

@app.route('/about-us')
def about_us():
    return render_template('about_us.html', title="About Us")

if __name__ == '__main__':
    app.run(debug=True)
