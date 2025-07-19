import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('eeg_sorted_by_task.csv')  # Update the path as needed

# Select numeric columns
numeric_cols = df.select_dtypes(include=['number']).columns

# Layout configuration
plt.style.use('ggplot')
num_cols = len(numeric_cols)
cols = 3
rows = (num_cols + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
axes = axes.flatten()

# Plot histograms with axis limits
for i, col in enumerate(numeric_cols):
    axes[i].hist(df[col], bins=30, edgecolor='black', color='tomato')
    axes[i].set_title(f'Histogram of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')
    axes[i].set_xlim(-7, 7)            # Set x-axis range
    axes[i].set_ylim(0, 2500)         # Set y-axis range

# # Hide unused subplots
# for i in range(len(numeric_cols), len(axes)):
#     fig.delaxes(axes[i])

plt.tight_layout()
plt.show()
