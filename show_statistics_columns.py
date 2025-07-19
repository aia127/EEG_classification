import pandas as pd

# Replace with your file path
file_path = 'merged_filtered_dataset.csv'

# Load data
df = pd.read_csv(file_path)
df=df.iloc[:,:16]
# Generate descriptive statistics for 16 electrodes
stats = df.describe().transpose()

# Save to CSV (optional)
stats.to_csv('column_statistics.csv')

# Display
print(stats)
