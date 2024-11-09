import pandas as pd
import numpy as np

# Seed for reproducibility
np.random.seed(42)

# Define the number of transactions
num_transactions = 100

# Generate random data for features
amount = np.random.uniform(10, 10000, num_transactions)  # Transaction amount (between 10 and 10,000 INR)
time = np.random.randint(0, 24, num_transactions)         # Time of transaction (0 to 23, hours of the day)
location = np.random.randint(0, 2, num_transactions)      # Location (0 = City, 1 = Rural)
device_info = np.random.randint(0, 3, num_transactions)   # Device Type (0 = Mobile, 1 = Tablet, 2 = Laptop)

# Generate the target variable: fraud detection (0 = Normal, 1 = Fraudulent)
# Set 2 random indices for fraudulent transactions
target = np.zeros(num_transactions)
fraud_indices = np.random.choice(num_transactions, 2, replace=False)
target[fraud_indices] = 1

# Create a pandas DataFrame
data = pd.DataFrame({
    'amount': amount,
    'time': time,
    'location': location,
    'device_info': device_info,
    'transaction_type': target  # 0 for normal, 1 for fraudulent
})

# Save to a CSV file
data.to_csv('upi_transactions.csv', index=False)

print("Dataset generated and saved to 'upi_transactions.csv'")