from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from src.utils.main_utils import load_object
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "final_model/model.pkl"

def load_model():
    """Load the trained model"""
    try:
        if os.path.exists(MODEL_PATH):
            model = load_object(MODEL_PATH)
            return model
        else:
            print(f"Model not found at {MODEL_PATH}")
            return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_model()

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction based on form data"""
    try:
        if model is None:
            return render_template('index.html', 
                                 prediction='Error: Model not loaded',
                                 error=True)
        
        # Get form data
        features = {
            'no_of_adults': int(request.form['no_of_adults']),
            'no_of_children': int(request.form['no_of_children']),
            'no_of_weekend_nights': int(request.form['no_of_weekend_nights']),
            'no_of_week_nights': int(request.form['no_of_week_nights']),
            'type_of_meal_plan': request.form['type_of_meal_plan'],
            'required_car_parking_space': int(request.form['required_car_parking_space']),
            'room_type_reserved': request.form['room_type_reserved'],
            'lead_time': int(request.form['lead_time']),
            'arrival_year': int(request.form['arrival_year']),
            'arrival_month': int(request.form['arrival_month']),
            'arrival_date': int(request.form['arrival_date']),
            'market_segment_type': request.form['market_segment_type'],
            'repeated_guest': int(request.form['repeated_guest']),
            'no_of_previous_cancellations': int(request.form['no_of_previous_cancellations']),
            'no_of_previous_bookings_not_canceled': int(request.form['no_of_previous_bookings_not_canceled']),
            'avg_price_per_room': float(request.form['avg_price_per_room']),
            'no_of_special_requests': int(request.form['no_of_special_requests'])
        }
        
        # Create DataFrame with features in correct order
        feature_order = [
            'no_of_adults', 'no_of_children', 'no_of_weekend_nights',
            'no_of_week_nights', 'type_of_meal_plan', 'required_car_parking_space',
            'room_type_reserved', 'lead_time', 'arrival_year', 'arrival_month',
            'arrival_date', 'market_segment_type', 'repeated_guest',
            'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled',
            'avg_price_per_room', 'no_of_special_requests'
        ]
        
        input_df = pd.DataFrame([features], columns=feature_order)
        
        # Make prediction
        prediction = model.predict(input_df)
        
        # Convert prediction to readable format
        result = 'Canceled' if prediction[0] == 1 else 'Not_Canceled'
        
        return render_template('index.html', prediction=result)
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('index.html', 
                             prediction=f'Error: {str(e)}',
                             error=True)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions (JSON)"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        
        # Create DataFrame
        input_df = pd.DataFrame([data])
        
        # Make prediction
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df) if hasattr(model, 'predict_proba') else None
        
        result = {
            'prediction': 'Canceled' if prediction[0] == 1 else 'Not_Canceled',
            'prediction_value': int(prediction[0])
        }
        
        if probability is not None:
            result['probability'] = {
                'not_canceled': float(probability[0][0]),
                'canceled': float(probability[0][1])
            }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
