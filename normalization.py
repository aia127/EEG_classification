import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load your data
df = pd.read_csv('eeg_sorted_by_task.csv')  # Change to your file path if needed

# Select only numeric columns to normalize
numeric_cols = df.select_dtypes(include=['number']).columns

# Initialize the scaler
scaler = MinMaxScaler()

# Apply normalization
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Save normalized data to a new CSV (optional)
df.to_csv('eeg_normalized.csv', index=False)

print("Normalization complete. Data saved to 'eeg_normalized.csv'.")
