import pandas as pd

# Replace with your file path
file_path = 'eeg_with_row_statistics_feature_uplifting.csv'

# Load data
df = pd.read_csv(file_path)

# Generate descriptive statistics
stats = df.describe().transpose()

# Save to CSV (optional)
stats.to_csv('with_genearated feature.csv')

# Display
print(stats)
