from flask import Flask, jsonify, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import yfinance as yf
app = Flask(__name__)

# Load data from CSV file
df = pd.read_csv('stock_data.csv')
for col in ['Open', 'High', 'Low', 'Close','Volume']:
    df[col] = df[col].str.replace(',', '').astype(float)

# Convert date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Extract year, month, and day from date column
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# Define features and target
features = ['Year', 'Month', 'Day', 'Open', 'High', 'Low', 'Volume']
target = 'Close'

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Define a function to predict stock prices
def predict_price(year, month, day, open_price, high_price, low_price, volume):
    input_data = [[year, month, day, open_price, high_price, low_price, volume]]
    return model.predict(input_data)[0]

# Define a function to compute model accuracy
def compute_accuracy():
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    accuracy = 1 - mse / y_test.var()
    return accuracy


def get_latest_data(ticker):
    stock_data = yf.download(ticker, period='1d')
    latest_data = stock_data.tail(1)
    latest_data['Date'] = latest_data.index.strftime('%Y-%m-%d')
    latest_data = latest_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    latest_data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    latest_data = latest_data.to_dict('records')
    return latest_data[0]

@app.route('/')
def index():
    # Compute model accuracy
    accuracy = compute_accuracy()
    latest_data = get_latest_data('AAPL')

    # Render the dashboard template
    return render_template('dashboard.html', accuracy=accuracy)

@app.route('/predict', methods=['POST'])
def predict():
    # Parse input data
    year = int(request.form['year'])
    month = int(request.form['month'])
    day = int(request.form['day'])
    open_price = float(request.form['open_price'])
    high_price = float(request.form['high_price'])
    low_price = float(request.form['low_price'])
    volume = float(request.form['volume'])

    # Make prediction
    price = predict_price(year, month, day, open_price, high_price, low_price, volume)

    # Return prediction as JSON response
    return jsonify({'price': price})

if __name__ == '__main__':
    app.run(debug=False)
