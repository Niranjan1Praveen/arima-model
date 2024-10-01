import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

PRICE_CSV_FILE_PATH = 'data/crop_prices_state_expanded.csv'

def load_data():
    """Load historical price data."""
    try:
        price_data = pd.read_csv(PRICE_CSV_FILE_PATH)
        return price_data
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {PRICE_CSV_FILE_PATH} was not found.")
    except Exception as e:
        raise Exception(f"An error occurred while loading data: {e}")

def predict_prices(crop_type, month, year, price_change=0):
    """Predict prices based on historical data and scenario."""
    price_data = load_data()
    
    # Filter historical data for the specified crop
    crop_data = price_data[price_data['crop_type'].str.strip().str.lower() == crop_type.strip().lower()].copy()
    
    if crop_data.empty:
        raise ValueError("No historical data available for the specified crop.")
    
    # Convert month_year to datetime and extract features
    crop_data['date'] = pd.to_datetime(crop_data['month_year'])
    crop_data['month'] = crop_data['date'].dt.month
    crop_data['year'] = crop_data['date'].dt.year
    
    # Features and target variable
    X = crop_data[['month', 'year']]
    y = crop_data['price']
    
    # Train a polynomial regression model
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LinearRegression
    
    poly = PolynomialFeatures(degree=3)
    model = make_pipeline(poly, LinearRegression())
    model.fit(X, y)
    
    # Predict future prices
    future_dates = pd.date_range(start=f'{year}-{month}-01', periods=30, freq='D')
    future_months = future_dates.month
    future_years = future_dates.year
    future_X = pd.DataFrame({'month': future_months, 'year': future_years})
    
    predicted_prices = model.predict(future_X)
    
    # Apply scenario changes
    predicted_prices += predicted_prices * price_change
    
    avg_price = predicted_prices.mean()
    
    future_data = [{'date': date.strftime('%Y-%m-%d'), 'price': price} for date, price in zip(future_dates, predicted_prices)]
    
    return crop_data['date'], crop_data['price'], future_data, avg_price

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        crop_type = data.get('crop_type')
        month_year = data.get('month_year')
        threshold_limit = float(data.get('threshold', 0))
        scenario = data.get('scenario', 'none')

        year, month = map(int, month_year.split('-'))

        price_change = 0
        if scenario == 'price_increase':
            price_change = 0.10  # Example: 10% increase
        elif scenario == 'price_decrease':
            price_change = -0.10  # Example: 10% decrease

        # Call the prediction function
        historical_dates, historical_prices, future_data, avg_price = predict_prices(crop_type, month, year, price_change)

        # Release price suggestion logic
        release_price_suggestion = None
        price_difference = None
        if avg_price >= threshold_limit:
            release_price_suggestion = avg_price * 1.02  # Add 2% to the average price
            price_difference = release_price_suggestion - avg_price

        # Construct the response data
        response = {
            'historical_dates': historical_dates.tolist(),
            'historical_prices': historical_prices.tolist(),
            'future_data': future_data,
            'avg_price': avg_price,
            'release_price_suggestion': release_price_suggestion,
            'price_difference': price_difference,
        }

        return jsonify(response)

    except ValueError as e:
        return jsonify({'message': str(e)}), 400
    except Exception as e:
        return jsonify({'message': 'An error occurred: {}'.format(str(e))}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
