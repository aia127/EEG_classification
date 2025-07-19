import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('eeg_sorted_by_task.csv')  # Update if needed

# Select only numeric columns
numeric_cols = df.select_dtypes(include=['number']).columns

# Set up plot style and size
plt.style.use('ggplot')
num_cols = len(numeric_cols)
cols = 3  # Number of plots per row
rows = (num_cols + cols - 1) // cols  # Ceiling division for subplot grid

# Create subplots
fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
axes = axes.flatten()

# Plot histograms
for i, col in enumerate(numeric_cols):
    axes[i].hist(df[col], bins=30, edgecolor='black')
    axes[i].set_title(f'Histogram of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

# Hide any unused subplots
for i in range(len(numeric_cols), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()
