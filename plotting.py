import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load the sorted dataset ---
df = pd.read_csv("eeg_iqr_filtered_rows.csv")
df_columns=[column for column in df.columns[:16]]
for column in df_columns:
    #Choosing electrode from the column to plot 
    electrode = column

    #Set up the plot figure
    plt.figure(figsize=(14, 7))
    sns.scatterplot(x=df.index, y=df[electrode], hue=df["task_type"], s=2, alpha=0.8)

    #Setting up the labels and titles
    plt.title(f"EEG Amplitude of {electrode} Over Time (Unsorted)", fontsize=14)
    plt.xlabel("Sample Index", fontsize=12)
    plt.ylabel("Amplitude", fontsize=12)
    plt.legend(title="Task Type")
    plt.grid(True)
    plt.ylim(-7,7)
    plt.tight_layout()

    #plot
    plt.show()
