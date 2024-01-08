import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import requests

from dotenv import load_dotenv
import os

# Load variables from .env file into the environment
load_dotenv()

APCA_API_KEY_ID = os.getenv('APCA_API_KEY_ID')
APCA_API_SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')

def Order(OrderType,qty,symbol):
    
  url = "https://paper-api.alpaca.markets/v2/orders"

  payload = {
      "symbol": symbol,
      "qty": qty,
      "notional": "string",
      "side": OrderType,
      "type": "market",
      "time_in_force": "day",
  }
  headers = {
      "APCA-API-KEY-ID":APCA_API_KEY_ID,
      "APCA-API-SECRET-KEY":APCA_API_SECRET_KEY,
      "accept": "application/json",
      "content-type": "application/json"
  }

  response = requests.post(url, json=payload, headers=headers)

  print(response.text)
  if response.status_code==200:
      return True 
  else: 
      return False

def getAccountDetail():
    url = "https://paper-api.alpaca.markets/v2/account"
    
    payload = {}
    headers = {
        "APCA-API-KEY-ID":APCA_API_KEY_ID,
        "APCA-API-SECRET-KEY":APCA_API_SECRET_KEY,
        "accept": "application/json",
        "content-type": "application/json"
    }
    
    response = requests.get(url, headers=headers, data=payload)
    print(response.text)
    return response.json()

def getBars(symbol, timeframe, start, end, adjustment):
    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars?timeframe={timeframe}&limit=1000&adjustment={adjustment}&feed=sip&sort=asc"
    print(url)
    headers = {
        "accept": "application/json",
        "APCA-API-KEY-ID":APCA_API_KEY_ID,
        "APCA-API-SECRET-KEY":APCA_API_SECRET_KEY,
    }
    
    response = requests.get(url, headers=headers)
    print(response.text)
    return response.json()

def getQuotes(symbol, limit):
    url = f"https://data.alpaca.markets/v1/quotes/{symbol}"
    
    payload = {
        "limit": limit,
    }
    headers = {
        "APCA-API-KEY-ID":APCA_API_KEY_ID,
        "APCA-API-SECRET-KEY":APCA_API_SECRET_KEY,
        "accept": "application/json",
        "content-type": "application/json"
    }
    
    response = requests.get(url, headers=headers, data=payload)
    print(response.text)
    return response.json()

def getQuotesIter(symbol):
    url = f"https://data.alpaca.markets/v1/last_quote/stocks/{symbol}"
    
    payload = {}
    headers = {
        "APCA-API-KEY-ID":APCA_API_KEY_ID,
        "APCA-API-SECRET-KEY":APCA_API_SECRET_KEY,
        "accept": "application/json",
        "content-type": "application/json"
    }
    
    response = requests.get(url, headers=headers, data=payload)
    print(response.text)
    return response.json()

def getTrades(symbol):
    url = f"https://data.alpaca.markets/v1/trades/{symbol}"
    
    payload = {}
    headers = {
        "APCA-API-KEY-ID":APCA_API_KEY_ID,
        "APCA-API-SECRET-KEY":APCA_API_SECRET_KEY,
        "accept": "application/json",
        "content-type": "application/json"
    }
    
    response = requests.get(url, headers=headers, data=payload)
    print(response.text)
    return response.json()

def getTradesIter(symbol):
    url = f"https://data.alpaca.markets/v1/last/stocks/{symbol}"
    
    payload = {}
    headers = {
        "APCA-API-KEY-ID":APCA_API_KEY_ID,
        "APCA-API-SECRET-KEY":APCA_API_SECRET_KEY,
        "accept": "application/json",
        "content-type": "application/json"
    }
    
    response = requests.get(url, headers=headers, data=payload)
    print(response.text)
    return response.json()

def getAssets(status, asset_class):
    
    url = "https://paper-api.alpaca.markets/v2/assets"

    payload = {
        "status": status,
        "asset_class": asset_class,
    }
    headers = {
        "APCA-API-KEY-ID":APCA_API_KEY_ID,
        "APCA-API-SECRET-KEY":APCA_API_SECRET_KEY,
        "accept": "application/json",
        "content-type": "application/json"
    }

    response = requests.get(url, headers=headers, data=payload)

    print(response.text)
    return response.json()

def getAsset(symbol):
    
    url = f"https://paper-api.alpaca.markets/v2/assets/{symbol}"

    payload = {}
    headers = {
        "APCA-API-KEY-ID":APCA_API_KEY_ID,
        "APCA-API-SECRET-KEY":APCA_API_SECRET_KEY,
        "accept": "application/json",
        "content-type": "application/json"
    }

    response = requests.get(url, headers=headers, data=payload)

    print(response.text)
    return response.json()

def getPosition(symbol):
    
    url = f"https://paper-api.alpaca.markets/v2/positions/{symbol}"

    payload = {}
    headers = {
        "APCA-API-KEY-ID":APCA_API_KEY_ID,
        "APCA-API-SECRET-KEY":APCA_API_SECRET_KEY,
        "accept": "application/json",
        "content-type": "application/json"
    }

    response = requests.get(url, headers=headers, data=payload)

    print(response.text)
    return response.json()


# Trading parameters
timeframe = "1Min"  # Minute timeframe for high-frequency trading
short_window = 50  # Short moving average window
long_window = 200  # Long moving average window
trade_frequency = 1000  # Number of trades per day

# Position sizing and risk management
risk_per_trade = 0.02  # Risk 2% of the account per trade
portfolio_value = float(getAccountDetail()['cash'])
print(portfolio_value)
risk_amount = portfolio_value * risk_per_trade

# Machine learning parameters
training_window = 20  # Window size for feature calculation
prediction_window = 5  # Number of future data points to predict

# Different machine learning models
model_for_strategy = RandomForestClassifier(n_estimators=100, random_state=42)
model_for_condor_vs_butterfly = SVC(kernel='linear', random_state=42)

# Setup logging
logging.basicConfig(filename='trading_log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Function to check if the short moving average crosses above the long moving average
def is_crossover(data):
    short_ma = data['close'].rolling(window=short_window).mean()
    long_ma = data['close'].rolling(window=long_window).mean()

    return short_ma.iloc[-1] > long_ma.iloc[-1] and short_ma.iloc[-2] <= long_ma.iloc[-2]

# Function to check if the short moving average crosses below the long moving average
def is_crossunder(data):
    short_ma = data['close'].rolling(window=short_window).mean()
    long_ma = data['close'].rolling(window=long_window).mean()

    return short_ma.iloc[-1] < long_ma.iloc[-1] and short_ma.iloc[-2] >= long_ma.iloc[-2]

# Function to extract features for machine learning
def extract_features(data):
    print(data)
    data['short_ma'] = data['close'].rolling(window=short_window).mean()
    data['long_ma'] = data['close'].rolling(window=long_window).mean()
    data['return'] = data['close'].pct_change()
    data['volatility'] = data['return'].rolling(window=training_window).std()
    data['target'] = np.where(data['close'].shift(-prediction_window) > data['close'], 1, 0)
    return data.dropna()

# Train the machine learning model for strategy prediction
def train_model_for_strategy(data):
    X = data[['short_ma', 'long_ma', 'return', 'volatility']]
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_for_strategy.fit(X_train, y_train)

    predictions = model_for_strategy.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    logging.info(f"Strategy Prediction Model Accuracy: {accuracy}")

# Train the machine learning model for Condor vs. Butterfly prediction
def train_model_for_condor_vs_butterfly(data):
    # Replace 'some_option_related_features' with actual option-related features
    X = data[['some_option_related_features']]
    y = data['target']  # Assuming you have a target variable (1 for Condor, 0 for Butterfly)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_for_condor_vs_butterfly.fit(X_train, y_train)

    predictions = model_for_condor_vs_butterfly.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    logging.info(f"Condor vs. Butterfly Model Accuracy: {accuracy}")

# Function to predict the best strategy
def predict_best_strategy(data):
    # Replace 'some_option_related_features' with actual option-related features
    features_for_condor_vs_butterfly = data[['some_option_related_features']]

    # Predict using the SVM model
    prediction = model_for_condor_vs_butterfly.predict(features_for_condor_vs_butterfly)[0]

    if prediction == 1:
        return 'iron_condor'
    else:
        return 'butterfly_spread'

# Function to check if Iron Condor strategy conditions are met
def is_iron_condor(data):
    # Implement Iron Condor conditions: Check if call and put options are out-of-the-money
    call_condition = data['strike_price_call'] > data['current_stock_price']
    put_condition = data['strike_price_put'] < data['current_stock_price']
    return call_condition and put_condition

# Function to check if Butterfly Spread strategy conditions are met
def is_butterfly_spread(data):
    # Implement Butterfly Spread conditions: Check if two call options have the same strike price
    call_condition_1 = data['strike_price_call_1'] == data['current_stock_price']
    call_condition_2 = data['strike_price_call_2'] == data['current_stock_price']
    return call_condition_1 and call_condition_2

# Function to execute Iron Condor strategy
def execute_iron_condor(symbol):
    logging.info(f"Iron Condor strategy executed for {symbol}")

# Function to execute Butterfly Spread strategy
def execute_butterfly_spread(symbol):
    logging.info(f"Butterfly Spread strategy executed for {symbol}")

# Main trading function
def trade():
    for symbol in symbols:
        # Initialize an empty DataFrame to store historical data
        historical_data = pd.DataFrame()
        # Define the time interval for each request (e.g., 100 days)
        interval = timedelta(days=100)

        for years_back in range(6, 0, -1):
            # Calculate start date based on the number of years back
            start_date = (datetime.now() - timedelta(days=years_back * 365)).strftime('%Y-%m-%d')
            current_date = datetime.strptime(start_date, '%Y-%m-%d')

            # Fetch historical data in chunks
            while current_date < datetime.now():
                # Calculate the end date for this chunk
                chunk_end_date = (current_date + interval).strftime('%Y-%m-%d')
                try:
                    # Fetch historical data for the current chunk
                    print(symbol, timeframe, current_date.strftime('%Y-%m-%d'),chunk_end_date)
                    chunk_data_response = getBars(symbol, timeframe, start=current_date.strftime('%Y-%m-%d'), end=chunk_end_date, adjustment='raw')
                    print("Fetched",chunk_data_response)
                    chunk_data = pd.DataFrame(chunk_data_response[symbol])
                    print(chunk_data)
                except Exception as e:
                    # Handle other exceptions that might occur during data fetching
                    logging.error(f"Error fetching data for symbol {symbol}: {e}")
                    continue

                # Concatenate the chunk data to the main DataFrame
                historical_data = pd.concat([historical_data, chunk_data])
                print(historical_data)
                # Move to the next chunk
                current_date += interval

            # Check if any data was fetched for the current year
            if not historical_data.empty:
                break  # Exit the outer loop if data was fetched

        # Extract features for machine learning
        features_data = extract_features(historical_data.copy())

        # Train the machine learning models
        train_model_for_strategy(features_data)
        train_model_for_condor_vs_butterfly(features_data)

        # Simulate high-frequency trading
        for _ in range(trade_frequency):
            # Make predictions using the machine learning model for strategy
            current_data = extract_features(historical_data.copy().tail(training_window))
            strategy_prediction = model_for_strategy.predict(current_data[['short_ma', 'long_ma', 'return', 'volatility']].iloc[-1].values.reshape(1, -1))[0]

            # Fine-tune strategies based on machine learning predictions
            strategies = []

            # Example strategies
            strategies.append({
                'name': 'Moving Average Crossover',
                'condition': is_crossover(historical_data),
                'action': 'buy',
            })

            strategies.append({
                'name': 'Moving Average Crossunder',
                'condition': is_crossunder(historical_data),
                'action': 'sell',
            })

            # Determine the best strategy using SVM prediction
            best_strategy = predict_best_strategy(current_data)

            # Execute trading actions based on strategies
            for strategy in strategies:
                if strategy['condition']:
                    if strategy['action'] == 'buy':
                        # Calculate position size based on risk
                        stop_loss_price = historical_data['low'].min()
                        position_size = int(risk_amount / (historical_data['close'].iloc[-1] - stop_loss_price))
                        Order('buy',position_size,symbol)
                        logging.info(f"Buy order executed for {symbol} - Quantity: {position_size}, Entry Price: {historical_data['close'].iloc[-1]}")

                    elif strategy['action'] == 'sell':
                        # Get current position and sell all shares
                        position = getPosition(symbol)
                        if position:
                            # Execute sell order
                            Order('sell',position.qty,symbol)
                            logging.info(f"Sell order executed for {symbol} - Quantity: {position.qty}, Exit Price: {historical_data['close'].iloc[-1]}")

                    # Implement options trading strategies based on machine learning predictions
                    if best_strategy == 'iron_condor' and is_iron_condor(current_data):
                        execute_iron_condor(symbol)

                    elif best_strategy == 'butterfly_spread' and is_butterfly_spread(current_data):
                        execute_butterfly_spread(symbol)

if __name__ == "__main__":
        # Assuming getAssets returns a list of dictionaries
    assets_list = getAssets(status='active', asset_class='us_equity')

    # Extracting 'symbol' from each dictionary
    symbols = [asset['symbol'] for asset in assets_list]

    # Printing the resulting list of symbols
    print(symbols)

    # Main trading loop
    while True:
        trade()
