import pandas as pd
# Compute median, min, and max for each row across EEG columns
df=pd.read_csv("eeg_iqr_filtered_rows.csv")
eeg_columns = df.select_dtypes(include='number').columns
row_medians = df[eeg_columns].median(axis=1)
row_mins = df[eeg_columns].min(axis=1)
row_maxs = df[eeg_columns].max(axis=1)

# Combine them into a new DataFrame
summary_df = pd.DataFrame({
    'median_value': row_medians,
    'min_value': row_mins,
    'max_value': row_maxs
})

# Concatenate the original DataFrame with the new summary columns
combined_df = pd.concat([df, summary_df], axis=1)

# Display the updated DataFrame to the user
combined_df.to_csv("eeg_with_row_statistics_feature_uplifting.csv", index=False)
