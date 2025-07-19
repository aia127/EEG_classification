import pandas as pd

#Load data
file_path = "merged_filtered_dataset.csv"  # 🔁 Change this to your EEG CSV file
df = pd.read_csv(file_path)

#16 columns are EEG channels ---
eeg_cols = df.columns[:16]
other_cols = df.columns[16:]

eeg_data = df[eeg_cols]
other_data = df[other_cols]

#Build a mask: True where values are within IQR per channel
iqr_mask = pd.DataFrame(index=eeg_data.index)

for col in eeg_cols:
    q1 = eeg_data[col].quantile(0.25)
    q3 = eeg_data[col].quantile(0.75)
    iqr_mask[col] = (eeg_data[col] >= q1) & (eeg_data[col] <= q3)

#Keep only rows where ALL EEG values are in their IQRs
rows_to_keep = iqr_mask.all(axis=1)
filtered_eeg = eeg_data[rows_to_keep]
filtered_other = other_data[rows_to_keep]

#Combine filtered data
final_df = pd.concat([filtered_eeg.reset_index(drop=True), filtered_other.reset_index(drop=True)], axis=1)

#Save result
final_df.to_csv("eeg_iqr_filtered_rows.csv", index=False)
print("✅ Saved to eeg_iqr_filtered_rows.csv")
