from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os

app = Flask(__name__)
CORS(app)

class DelinquencyPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def load_and_preprocess_data(self, filepath='delinquency_data.csv'):
        try:
            # Check if file exists
            if not os.path.exists(filepath):
                print(f"Data file {filepath} not found. Creating sample data...")
                # Create sample data if file doesn't exist
                sample_data = {
                    'Age': [25, 35, 45, 55, 28, 40, 50, 33, 38, 48],
                    'Income': [30000, 50000, 70000, 90000, 35000, 60000, 80000, 45000, 55000, 75000],
                    'Credit_Score': [650, 700, 750, 800, 620, 680, 770, 660, 720, 780],
                    'Missed_Payments': [1, 2, 0, 1, 3, 1, 0, 2, 1, 0],
                    'Debt_to_Income_Ratio': [0.3, 0.4, 0.2, 0.35, 0.5, 0.25, 0.15, 0.45, 0.33, 0.22],
                    'Delinquent': [1, 1, 0, 0, 1, 0, 0, 1, 0, 0]
                }
                df = pd.DataFrame(sample_data)
                df.to_csv(filepath, index=False)
                print(f"Sample data created and saved to {filepath}")
            
            # Load data
            df = pd.read_csv(filepath)
            
            # Basic preprocessing
            df.dropna(inplace=True)
            
            return df
        except Exception as e:
            print(f"Data loading error: {e}")
            return None
    
    def train_model(self, df):
        try:
            # Select features and target
            features = ['Age', 'Income', 'Credit_Score', 'Missed_Payments', 'Debt_to_Income_Ratio']
            
            # Ensure all required columns exist
            if not all(col in df.columns for col in features + ['Delinquent']):
                print("Missing required columns")
                return False
            
            X = df[features]
            y = df['Delinquent']
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Train Random Forest model
            self.model = RandomForestClassifier(
                n_estimators=100, 
                random_state=42
            )
            self.model.fit(X_train, y_train)
            
            # Model evaluation
            train_accuracy = self.model.score(X_train, y_train)
            test_accuracy = self.model.score(X_test, y_test)
            
            print(f"Model Training Completed")
            print(f"Train Accuracy: {train_accuracy:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")
            
            self.is_trained = True
            return True
        
        except Exception as e:
            print(f"Model training error: {e}")
            return False
    
    def predict_delinquency(self, age, income, credit_score, missed_payments, debt_to_income_ratio):
        try:
            # Prepare input data
            input_data = np.array([
                [age, income, credit_score, missed_payments, debt_to_income_ratio]
            ])
            
            # Scale input data
            input_scaled = self.scaler.transform(input_data)
            
            # Predict
            if self.model:
                prediction = self.model.predict(input_scaled)[0]
                proba = self.model.predict_proba(input_scaled)[0]
                
                return {
                    'is_delinquent': bool(prediction),
                    'risk_probability': float(proba[1]),
                    'confidence': float(np.max(proba))
                }
            
            # Fallback rule-based prediction
            return self._rule_based_prediction(
                credit_score, missed_payments, debt_to_income_ratio
            )
        
        except Exception as e:
            print(f"Prediction error: {e}")
            return self._rule_based_prediction(
                credit_score, missed_payments, debt_to_income_ratio
            )
    
    def _rule_based_prediction(self, credit_score, missed_payments, debt_to_income_ratio):
        """Fallback rule-based prediction method"""
        risk_factors = 0
        
        if credit_score < 600:
            risk_factors += 1
        if missed_payments > 2:
            risk_factors += 1
        if debt_to_income_ratio > 0.4:
            risk_factors += 1
        
        is_delinquent = risk_factors >= 2
        
        return {
            'is_delinquent': is_delinquent,
            'risk_probability': risk_factors / 3,
            'confidence': 0.7,
            'method': 'rule_based'
        }

# Initialize predictor
delinquency_predictor = DelinquencyPredictor()

# Serve static files
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('.', path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Extract features
        age = float(data.get('age'))
        income = float(data.get('income'))
        credit_score = float(data.get('creditScore'))
        missed_payments = int(data.get('missedPayments'))
        debt_to_income_ratio = float(data.get('debtToIncomeRatio'))
        
        # Make prediction
        result = delinquency_predictor.predict_delinquency(
            age, income, credit_score, 
            missed_payments, debt_to_income_ratio
        )
        
        # Add human-readable messages
        if result['is_delinquent']:
            result['message'] = "High risk of financial delinquency detected."
            result['recommendation'] = "Recommended: Conduct further financial review"
        else:
            result['message'] = "Low risk of financial delinquency."
            result['recommendation'] = "Customer appears financially stable"
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/train', methods=['POST'])
def train_model():
    try:
        # Load data
        df = delinquency_predictor.load_and_preprocess_data()
        
        if df is None:
            return jsonify({'error': 'Failed to load data'}), 500
        
        # Train model
        success = delinquency_predictor.train_model(df)
        
        return jsonify({
            'trained': success,
            'message': 'Model trained successfully' if success else 'Training failed'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Optionally train on startup with sample data
    print("Loading sample data and initializing model...")
    df = delinquency_predictor.load_and_preprocess_data()
    if df is not None:
        delinquency_predictor.train_model(df)
    app.run(debug=True, port=5000)