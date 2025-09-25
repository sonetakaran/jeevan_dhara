# app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load your trained model
model = joblib.load('water_risk_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        df = pd.DataFrame([data])
        
        # ✅ CHANGED: Use the exact feature names the model was trained with
        features = ['ph', 'turbidity_ntu', 'temperature_c', 'dissolved_oxygen_mg_l']
        df = df[features]
        
        prediction = model.predict(df)
        
        return jsonify({'risk_prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)