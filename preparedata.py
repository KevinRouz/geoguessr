import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV file
data = pd.read_csv('processed_coords.csv', names=['latitude', 'longitude', 'city'])

# Define paths
image_dir = 'images/'
output_dir = 'organized_data/'

# Create folders for each city
cities = data['city'].unique()
for city in cities:
    os.makedirs(os.path.join(output_dir, 'train', city), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val', city), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test', city), exist_ok=True)

# Split data into train, val, and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['city'])
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42, stratify=train_data['city'])

# Function to move files
def move_files(data, subset):
    for index, row in data.iterrows():
        filename = f"{index + 1}_{row['city']}.jpg"
        src_path = os.path.join(image_dir, filename)
        dst_path = os.path.join(output_dir, subset, row['city'], filename)
        shutil.copy(src_path, dst_path)

# Move the files to train, val, test folders
move_files(train_data, 'train')
move_files(val_data, 'val')
move_files(test_data, 'test')

print("Data organized into train, val, and test sets.")