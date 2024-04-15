from flask import Flask, request, jsonify, render_template, url_for
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = load_model('model_epoch_best.h5')

class_names = {
    0: 'Benign keratosis',
    1: 'Melanocytic nevus',
    2: 'Dermatofibroma',
    3: 'Melanoma',
    4: 'Vascular lesion',
    5: 'Basal cell carcinoma',
    6: 'Actinic Keratosis'
}

@app.route('/')
def form():
    return render_template('upload_form.html')

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CONFIDENCE_THRESHOLD = 0.8

os.makedirs(os.path.join(app.root_path, UPLOAD_FOLDER), exist_ok=True)

@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = secure_filename(file.filename)
        img = Image.open(file.stream)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img.save(filepath) 

        # Preprocessing the image so that it matches the training input
        img = img.resize((224, 224))
        img_array = np.expand_dims(np.array(img), axis=0)

        predictions = model.predict(img_array)
        class_id = np.argmax(predictions, axis=-1)
        accuracy = np.max(predictions)

        # Creating a dictionary to hold the information to pass to the template
        template_data = {
            'class_name': '',
            'additional_info': '',
            'image_url': url_for('static', filename='uploads/' + filename)
        }        

        # Checking if the confidence is above the threshold
        if accuracy < CONFIDENCE_THRESHOLD:
            template_data['class_name'] = "No Disease Detected"
            template_data['additional_info'] = "The model is not confident in its assessment, and the skin may be healthy or may be having a disease other than the ones on which the model is trained on. Please consult a professional for a definitive diagnosis."
        else:
            class_name = class_names.get(int(class_id), "Unknown class")
            additional_info = get_additional_info(class_name)
            template_data.update({
                'class_name': class_name,
                'accuracy': accuracy,
                'additional_info': additional_info
            })

        return render_template('results.html', **template_data)

def get_additional_info(disease_name):
    disease_info = {
        'Benign keratosis': 'It is a generic class that includes seborrheic keratoses, lichen-planus-like keratoses, and solar lentigo. The three subgroups may look different dermatoscopically, but they are grouped as they are similar biologically and often reported under the same generic term histopathologically.',
        'Melanocytic nevus': 'Skin lesions are benign neoplasms of melanocytes and appear in various shapes and sizes. From a dermatoscopic standpoint, the variants may differ dramatically.',
        'Dermatofibroma': 'This skin lesion is either benign growth or an inflammatory response to minor trauma.',
        'Melanoma': 'Melanoma is a cancerous tumor that develops from melanocytes and can take many forms. It can be treated with a primary surgical procedure if caught early enough.',
        'Vascular lesion': 'Cherry antifoams, angiokeratomas, and pyogenic granulomas are examples of benign or malignant angiomas.',
        'Basal cell carcinoma': 'A type of epithelial skin cancer that rarely spreads, but if left untreated, it might become aggressive and relapse.',
        'Actinic Keratosis': 'Non-invasive type of squamous cell carcinoma that can be treated locally without surgery.'
    }
    return disease_info.get(disease_name, 'No additional information available.')    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
