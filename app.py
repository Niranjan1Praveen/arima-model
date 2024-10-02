from flask import Flask, request, jsonify
from flask_cors import CORS  
import pandas as pd
import os
from statsmodels.tsa.arima.model import ARIMA

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

data_path = os.path.join(os.getcwd(), 'data', 'cropPrices.csv')
crop_prices = pd.read_csv(data_path)

def predict_prices(filtered_data):
    try:
        filtered_data.loc[:, 'Arrival_Date'] = pd.to_datetime(filtered_data['Arrival_Date'], format='%d-%m-%Y')
        filtered_data = filtered_data.sort_values('Arrival_Date')
        
        if len(filtered_data) < 2:
            raise ValueError("Not enough data to make predictions")

        price_series = filtered_data['Modal Price'].astype(float).values
        
        model = ARIMA(price_series, order=(5, 1, 0)) 
        model_fit = model.fit()

        predictions = model_fit.forecast(steps=75)
        
        last_date = filtered_data['Arrival_Date'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=75, freq='D')  

        predicted_prices = [{'date': date.strftime('%Y-%m-%d'), 'predictedPrice': float(price)} 
                            for date, price in zip(future_dates, predictions)]
        return predicted_prices
    except Exception as e:
        return {'error': str(e)}

@app.route('/', methods=['POST'])
def index():
    data = request.get_json()
    crop = data.get('crop', '').lower().strip()
    state = data.get('state', '').lower().strip()
    district = data.get('district', '').lower().strip()

    if not crop or not state or not district:
        return jsonify({'error': 'Please provide all required inputs'}), 400

    filtered_data = crop_prices[
        (crop_prices['Commodity'].str.lower().str.contains(crop, na=False)) & 
        (crop_prices['State'].str.lower().str.contains(state, na=False)) & 
        (crop_prices['District'].str.lower().str.contains(district, na=False))
    ]

    if filtered_data.empty:
        return jsonify({'error': 'No data available for the specified crop, state, and district'}), 404

    # Predict prices
    predicted_prices = predict_prices(filtered_data)

    if 'error' in predicted_prices:
        return jsonify({'error': predicted_prices['error']}), 500

    # Send results as JSON response
    return jsonify(predictions=predicted_prices)

if __name__ == '__main__':
    app.run()
