import os
import matplotlib.pyplot as plt
from data_loader import StreetViewDataset

# Set parameters
data_dir = 'organized_data/train'  # Adjust this path as necessary
csv_file = 'processed_coords.csv'   # Adjust this path as necessary

# Create an instance of the dataset
dataset = StreetViewDataset(data_dir, csv_file)

# Check the length of the dataset
print(f"Total number of batches: {len(dataset)}")

# Check total number of images 
print(f"Total number of images: {len(dataset.image_paths)}")

# Test loading a single batch
images, labels = dataset[0]  # Load the first batch

# Print the shape of the images and labels
print(f"Images shape: {images.shape}")
print(f"Labels shape: {labels.shape}")

for i in range(len(dataset.image_paths)):
    image_path = dataset.image_paths[i]
    lat, long, city_name = dataset.labels[i]
    
    # Print the information
    print(f"Image: {os.path.basename(image_path)}, Latitude: {lat}, Longitude: {long}, City: {city_name}")

# Prepare to store one image from each city
city_images = {}
for img_path, (lat, long, city_name) in zip(dataset.image_paths, dataset.labels):
    if city_name not in city_images:
        city_images[city_name] = img_path  # Store the first image found for each city

# Visualize one image from each city
num_cities = len(city_images)
plt.figure(figsize=(15, 5))
for i, (city_name, img_path) in enumerate(city_images.items()):
    img = plt.imread(img_path)  # Read the image
    plt.subplot(1, num_cities, i + 1)
    plt.imshow(img.astype('uint8'))  # Convert to uint8 for display
    plt.title(f"City: {city_name}")
    plt.axis('off')

plt.show()