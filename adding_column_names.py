import pandas as pd

# Example: If you just read a CSV without headers
df = pd.read_csv('merged_filtered_dataset.csv', header=None)  # Avoids using first row as header

# Define your new column names
column_names = [
    'FC5', 'F3', 'Fz', 'F4', 'FC6',
    'C5', 'C3', 'Cz', 'C4', 'C6',
    'CP5', 'CP3', 'CP1', 'CP2', 'CP4', 'CP6',
    'subject_id','task_type'
]

# Assign them to the DataFrame
df.columns = column_names

# Done! Save if needed
df.to_csv('merged_filtered_dataset.csv', index=False)

