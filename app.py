from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load Models
heart_model = pickle.load(open('models/heart_disease_model.sav', 'rb'))
diabetes_model = pickle.load(open('models/diabetes_model.sav', 'rb'))
parkinsons_model = pickle.load(open('models/parkinsons_model.sav', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')  # Main page with options

@app.route('/predict/heart', methods=['GET', 'POST'])
def predict_heart():
    if request.method == 'POST':
        try:
            # Collect data from the form and convert to float
            age = float(request.form['age'])
            sex = float(request.form['sex'])
            cp = float(request.form['cp'])
            trestbps = float(request.form['trestbps'])
            chol = float(request.form['chol'])
            fbs = float(request.form['fbs'])
            restecg = float(request.form['restecg'])
            thalach = float(request.form['thalach'])
            exang = float(request.form['exang'])
            oldpeak = float(request.form['oldpeak'])
            slope = float(request.form['slope'])
            ca = float(request.form['ca'])
            thal = float(request.form['thal'])
            
            # Feature array
            features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
            
            # Prediction
            prediction = heart_model.predict(features)[0]
            
            # Output result
            result = "Heart Disease" if prediction == 1 else "No Heart Disease"
            return render_template('result.html', result=result, model_type='heart')
        
        except ValueError:
            # Handle invalid input or conversion errors
            error_message = "Please ensure all inputs are valid numeric values."
            return render_template('heart_form.html', error=error_message)

    return render_template('heart_form.html')  # Return the form page if not POST

@app.route('/predict/diabetes', methods=['GET', 'POST'])
def predict_diabetes():
    if request.method == 'POST':
        # Collect data from the form
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['blood_pressure'])
        skin_thickness = float(request.form['skin_thickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        diabetes_pedigree_function = float(request.form['diabetes_pedigree_function'])
        age = float(request.form['age'])
        
        # Feature array
        features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
        
        # Prediction
        prediction = diabetes_model.predict(features)[0]
        print(prediction)
        result = "Diabetic" if prediction == 1 else "Non-Diabetic"
        return render_template('result.html', result=result, model_type='diabetes')
    
    return render_template('diabetes_form.html')  # Diabetes form page

@app.route('/predict/parkinsons', methods=['GET', 'POST'])
def predict_parkinsons():
    if request.method == 'POST':
        # Collect data from form fields
        data = [
            float(request.form['bmi']),
            int(request.form['gender']),
            int(request.form['ethnicity']),
            float(request.form['alcohol_consumption']),
            float(request.form['physical_activity']),
            float(request.form['diet_quality']),
            float(request.form['sleep_quality']),
            int(request.form['family_history']),
            int(request.form['traumatic_brain_injury']),
            int(request.form['hypertension']),
            int(request.form['diabetes']),
            int(request.form['depression']),
            int(request.form['stroke']),
            float(request.form['systolic_bp']),
            float(request.form['diastolic_bp']),
            float(request.form['cholesterol_total']),
            float(request.form['cholesterol_ldl']),
            float(request.form['cholesterol_hdl']),
            float(request.form['cholesterol_triglycerides']),
            float(request.form['updrs']),
            float(request.form['moca']),
            float(request.form['functional_assessment']),
            int(request.form['tremor']),
            int(request.form['rigidity']),
            int(request.form['bradykinesia']),
            int(request.form['postural_instability']),
            int(request.form['speech_problems']),
            int(request.form['sleep_disorders']),
            int(request.form['constipation'])
        ]
        
        # Convert to a numpy array for the model
        data = np.array(data).reshape(1, -1)

        # Make the prediction
        prediction = parkinsons_model.predict(data)

        # Return result
        if prediction[0] == 1:
            result = "Parkinson's Disease Detected"
        else:
            result = "No Parkinson's Disease Detected"

        return render_template('result.html', result=result, model_type='parkinsons')

    return render_template('parkinsons_form.html')

@app.route('/about')
def about():
    return render_template('about.html')  # About page

# if __name__ == '__main__':
#     app.run(debug=True)
