import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

def predict_prices(filtered_data):
    filtered_data['Date'] = pd.to_datetime(filtered_data['Date'])
    filtered_data = filtered_data.sort_values('Date')
    
    price_series = filtered_data['Modal Price'].astype(float).values
    
    # model = ARIMA(price_series, order=(5, 1, 0))  (p,d,q) parameters
    # model_fit = model.fit()

    model = SARIMAX(price_series, order=(5, 1, 0), seasonal_order=(1, 1, 1, 12))  # Adjust seasonal_order as needed
    model_fit = model.fit(disp=False)

    predictions = model_fit.forecast(steps=75)
    
    # predictions = model_fit.forecast(steps=30)
    
    last_date = filtered_data['Date'].max()
    future_dates = pd.date_range(start=last_date, periods=30, freq='D')

    predicted_prices = [{'date': str(date.date()), 'predictedPrice': float(price)} for date, price in zip(future_dates, predictions)]
    print(predicted_prices)
    return predicted_prices
