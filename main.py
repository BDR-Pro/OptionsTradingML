import csv
import requests
import time
import pandas as pd

# Constants
API_KEY = 'YOUR_API_KEY'
API_SECRET = 'YOUR_API_SECRET'
BASE_URL = 'https://data.alpaca.markets/v2'
CSV_FILE = 'assets.csv'  # Path to your CSV file


from dotenv import load_dotenv
import os

# Load variables from .env file into the environment
load_dotenv()

APCA_API_KEY_ID = os.getenv('APCA_API_KEY_ID')
APCA_API_SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')


# Headers for authentication
headers = {
    'APCA-API-KEY-ID': APCA_API_KEY_ID,
    'APCA-API-SECRET-KEY': APCA_API_SECRET_KEY
}

def fetch_stock_data(ticker,page_token=""):
    """
    Fetch stock data for a given ticker
    """
    if page_token:
        page_token=f"page_token={page_token}&"
    url = f"{BASE_URL}/stocks/bars?symbols={ticker}&timeframe=1Min&start=2002-01-03T00%3A00%3A00Z&end=2024-01-14T00%3A00%3A00Z&limit=10000&adjustment=raw&feed=sip&{page_token}sort=asc"

    response = requests.get(url, headers=headers)
    print(response)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch data for {ticker}")
        return None

# Read tickers from CSV
tickers = []
with open(CSV_FILE, mode='r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        tickers.append(row[0])

# Fetch data for each ticker
data = []
for index, ticker in enumerate(tickers):
    try:
        print(f"ticker: {ticker} No.{index+1} out of {len(tickers)}")
        print(f"Completed: {round((index+1)/len(tickers)*100, 2)}%", end="\r")
        stock_data = fetch_stock_data(ticker)
        print(stock_data)
        if stock_data['bars']=={}:
            continue
        if stock_data:
            data.append(stock_data)
        if stock_data['next_page_token']:
            while stock_data['next_page_token']:
                stock_data = fetch_stock_data(ticker, stock_data['next_page_token'])
                if stock_data:
                    data.append(stock_data)
        time.sleep(1)  # To respect rate limits
    except Exception as e:
        print(e)
        continue

# Convert to DataFrame for better visualization and manipulation
df = pd.DataFrame(data)
# Save the data to a new CSV file
df.to_csv('stock_data.csv', index=False)
