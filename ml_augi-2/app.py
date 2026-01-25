from flask import Flask, request, render_template, jsonify, send_file
from src.exception import CustomException
from src.logger import logging
import os
import pandas as pd
import sys

from src.pipelines.predict_pipeline import PredictPipelines
from src.pipelines.training_pipeline import ModelTrainingPipeline

application = Flask(__name__)
app = application

# Enable logging to console
import logging as base_logging
console_handler = base_logging.StreamHandler(sys.stdout)
console_handler.setLevel(base_logging.INFO)
formatter = base_logging.Formatter('[ %(asctime)s ] %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
base_logging.getLogger().addHandler(console_handler)

# Create upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/train', methods=['GET', 'POST'])
def train_model():
    if request.method == 'GET':
        return render_template('train.html')

    try:
        # Check if file is uploaded
        if 'file' not in request.files:
            return render_template('train.html', result='Error: No file uploaded')

        file = request.files['file']

        if file.filename == '':
            return render_template('train.html', result='Error: No file selected')

        # Save uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        logging.info(f"File uploaded: {file_path}")

        # Run training pipeline
        logging.info("Starting training pipeline...")
        pipeline = ModelTrainingPipeline()
        score = pipeline.run_pipeline(file_path=file_path)

        result_message = f"Training completed successfully! Model accuracy: {score}"
        logging.info(result_message)
        print(f"\n{'='*80}")
        print(f"✅ TRAINING SUCCESSFUL!")
        print(f"Model Accuracy: {score}")
        print(f"{'='*80}\n")

        return render_template('train.html', result=result_message)

    except Exception as e:
        logging.error(f"Error in training: {str(e)}")
        return render_template('train.html', result=f'Error: {str(e)}')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('prediction.html')

    try:
        # Collect all features from form (cancer dataset - 23 features)
        features_dict = {
            'Age': int(request.form.get('Age')),
            'Gender': int(request.form.get('Gender')),
            'Air Pollution': int(request.form.get('Air Pollution')),
            'Alcohol use': int(request.form.get('Alcohol use')),
            'Dust Allergy': int(request.form.get('Dust Allergy')),
            'OccuPational Hazards': int(request.form.get('OccuPational Hazards')),
            'Genetic Risk': int(request.form.get('Genetic Risk')),
            'chronic Lung Disease': int(request.form.get('chronic Lung Disease')),
            'Balanced Diet': int(request.form.get('Balanced Diet')),
            'Obesity': int(request.form.get('Obesity')),
            'Smoking': int(request.form.get('Smoking')),
            'Passive Smoker': int(request.form.get('Passive Smoker')),
            'Chest Pain': int(request.form.get('Chest Pain')),
            'Coughing of Blood': int(request.form.get('Coughing of Blood')),
            'Fatigue': int(request.form.get('Fatigue')),
            'Weight Loss': int(request.form.get('Weight Loss')),
            'Shortness of Breath': int(request.form.get('Shortness of Breath')),
            'Wheezing': int(request.form.get('Wheezing')),
            'Swallowing Difficulty': int(request.form.get('Swallowing Difficulty')),
            'Clubbing of Finger Nails': int(request.form.get('Clubbing of Finger Nails')),
            'Frequent Cold': int(request.form.get('Frequent Cold')),
            'Dry Cough': int(request.form.get('Dry Cough')),
            'Snoring': int(request.form.get('Snoring'))
        }

        # Create DataFrame
        final_new_data = pd.DataFrame([features_dict])
        logging.info(f"Input features shape: {final_new_data.shape}")
        logging.info(f"Input features:\n{final_new_data.to_string()}")

        # Make prediction
        predict_pipeline = PredictPipelines(request=request)
        pred = predict_pipeline.predict(final_new_data)

        logging.info(f"Raw prediction: {pred}")
        logging.info(f"Prediction type: {type(pred[0])}")

        # The prediction is already converted to label by LabelEncoder
        result = pred[0] if isinstance(pred[0], str) else str(pred[0])

        logging.info(f"Final prediction result: {result}")
        print(f"\n{'='*80}")
        print(f"🔮 PREDICTION RESULT: {result}")
        print(f"{'='*80}\n")

        return render_template('prediction.html', final_result=result)

    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        return render_template('prediction.html', final_result=f'Error: {str(e)}')
    

if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True, port=5050)