import pandas as pd
import os
import glob

# Path to the main 'Dataset' folder
dataset_path = r'C:\dit\Machine learning SS25\machine_learning_exercises_git_repo\Mendeley_dataset_ML\Dataset'

all_dfs = []

# Patterns to exclude
exclude_patterns = ['I1', 'I8', 'M1', 'M8']

# Loop through each subject folder
for subject_folder in os.listdir(dataset_path):
    subject_path = os.path.join(dataset_path, subject_folder)
    
    if os.path.isdir(subject_path):
        csv_files = glob.glob(os.path.join(subject_path, '*.csv'))

        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            
            # Skip files that contain any of the excluded patterns
            if any(pattern in filename for pattern in exclude_patterns):
                continue

            df = pd.read_csv(csv_file)
            
            # Add subject_id column
            df['subject_id'] = subject_folder
            
            # Determine task_type
            if 'I' in filename:
                df['task_type'] = 'imaginary'
            elif 'M' in filename:
                df['task_type'] = 'action'
            

            all_dfs.append(df)

# Merge and save
merged_df = pd.concat(all_dfs, ignore_index=True)
merged_df.to_csv('merged_filtered_dataset.csv', index=False)
