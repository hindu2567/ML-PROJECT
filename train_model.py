import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
data = pd.read_csv(r'data\upi_transactions.csv')

# Prepare feature columns and target variable
X = data[['amount', 'time', 'location', 'device_info']]
y = data['transaction_type']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
with open('models/fraud_detection_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved!")
