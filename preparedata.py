import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV file
print("Loading 'processed_coords.csv'...")
data = pd.read_csv('processed_coords.csv', names=['latitude', 'longitude', 'city'])
print(f"Loaded {len(data)} records from 'processed_coords.csv'.")

# Define paths
image_dir = 'images/'
output_dir = 'organized_data/'

# Create folders for each city
print("Creating folders for each city...")
cities = data['city'].unique()
for city in cities:
    os.makedirs(os.path.join(output_dir, 'train', city), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val', city), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test', city), exist_ok=True)
print(f"Folders created for cities: {', '.join(cities)}")

# Split data into train, val, and test sets
print("Splitting data into train, validation, and test sets...")
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['city'])
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42, stratify=train_data['city'])
print(f"Data split complete: {len(train_data)} train records, {len(val_data)} validation records, {len(test_data)} test records.")

# Function to move files
def move_files(data, subset):
    print(f"Moving {len(data)} files to the {subset} folder...")
    for index, row in data.iterrows():
        filename = f"{index + 1}_{row['city']}.jpg"
        src_path = os.path.join(image_dir, filename)
        dst_path = os.path.join(output_dir, subset, row['city'], filename)
        
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"File not found: {src_path}")

    print(f"Finished moving files to {subset}.")

# Move the files to train, val, test folders
move_files(train_data, 'train')
move_files(val_data, 'val')
move_files(test_data, 'test')

print("Data organized into train, val, and test sets.")
