import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

# --- Parameters ---
file_path = "merged_filtered_dataset.csv"  # <-- Change this to your file
fs = 125  # Sampling frequency in Hz
order = 4  # Filter order
lowcut = 7.0
highcut = 31.0



# --- Bandpass Filter Function ---
def bandpass_filter(data, lowcut, highcut, fs, order=8):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# --- Load EEG Data ---
df = pd.read_csv(file_path)

# --- Apply Filter Only to EEG Channels (Assume first 16 columns are EEG) ---
eeg_cols = df.columns[:16]
other_cols = df.columns[16:]

filtered_eeg = df[eeg_cols].apply(lambda x: bandpass_filter(x.values, lowcut, highcut, fs), axis=0)

# --- Recombine Filtered EEG + Unchanged Other Columns ---
filtered_df = pd.concat([filtered_eeg, df[other_cols].reset_index(drop=True)], axis=1)

# --- Save Output ---
filtered_df.to_csv("eeg_filtered_8_30hz.csv", index=False)
print("Filtered file saved as eeg_filtered_8_30hz.csv")
