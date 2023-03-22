import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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

# Evaluate the model on the test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean squared error: {mse:.2f}")
