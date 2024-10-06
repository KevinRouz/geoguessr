import requests
from tqdm import tqdm
import os
import json
from random import randint
import argparse
from dotenv import load_dotenv
import sys
import re

load_dotenv()
api_key = os.getenv("GOOGLE_MAPS_API_KEY")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", help="The folder full of addresses per city to read and extract GPS coordinates from (defaults to: cities/)", default='cities/', type=str)
    parser.add_argument("--output", help="The output folder where the images will be stored, (defaults to: images/)", default='images/', type=str)
    parser.add_argument("--icount", help="The amount of images to pull (defaults to 50)", default=50, type=int)
    return parser.parse_args()


def load_processed_coords(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return list(tuple(line.strip().split(',')) for line in f) 
    return list()

def save_processed_coords(file_path, processed_coords):
    with open(file_path, 'w') as f:
        for coord in processed_coords:
            f.write(f'{coord[0]},{coord[1]},{coord[2]}\n') 
            
def get_next_image_index(output_folder):
    existing_files = os.listdir(output_folder)
    jpg_files = [f for f in existing_files if f.endswith('.jpg')]
    
    if not jpg_files:
        return 1 #1 indexed to match csv file rows
    
    # Extract the numeric part of the filenames (assuming '<city>_X.jpg')
    indexes = [int(re.search(r'\d+', f).group()) for f in jpg_files]
    
    return max(indexes) + 1  # Continue from the highest number

args = get_args()
url = 'https://maps.googleapis.com/maps/api/streetview'
cities = []

def load_cities():
    for city_file in os.listdir(args.cities):
        city_name = os.path.splitext(city_file)[0]  # Get the city name from the file name
        with open(os.path.join(args.cities, city_file)) as f:
            coordinates = []
            print(f'Loading {city_file} addresses...')
            for line in tqdm(f):
                data = json.loads(line)
                coordinates.append(data['geometry']['coordinates'])
            cities.append((city_name, coordinates))  # Store city name with coordinates

def main():
    os.makedirs(args.output, exist_ok=True)
    
    load_cities()
    processed_coords = load_processed_coords('processed_coords.csv')  # Load processed coordinates
    processed_coords_set = set(processed_coords)  # Create a set for quick lookups, but processed_coords remains a list to maintain orig order
    
    next_image_index = get_next_image_index(args.output)
    
    cities_count = [0] * len(cities)
    successful_images = []
    successful_images_count = 0

    with tqdm(total=args.icount, desc="Retrieving Images", unit="image") as pbar:
        while successful_images_count < args.icount:
            city_index = randint(0, len(cities) - 1)
            city_name, city_coordinates = cities[city_index]
            
            while True:
                addressLoc = city_coordinates[randint(0, len(city_coordinates) - 1)]
                if (addressLoc[1], addressLoc[0], city_name) not in processed_coords_set:
                    break
            
            params = {
                'key': api_key,
                'size': '640x640',
                'location': f"{addressLoc[1]},{addressLoc[0]}",
                'heading': str((randint(0, 3) * 90) + randint(-15, 15)),
                'pitch': '20',
                'fov': '90'
            }
            
            response = requests.get(url, params)
            
            if response.status_code == 200:
                image_content = response.content
                
                if len(image_content) > 9 * 1024:
                    successful_images.append((image_content, city_name, next_image_index))
                    
                    # Add the coordinates to both the list and set
                    processed_coords.append((addressLoc[1], addressLoc[0], city_name))
                    processed_coords_set.add((addressLoc[1], addressLoc[0], city_name))
                    
                    print(next_image_index, addressLoc[1], addressLoc[0], city_name)
                    
                    cities_count[city_index] += 1
                    successful_images_count += 1
                    next_image_index += 1
                    
                    pbar.update(1)
                else:
                    print(f"Invalid image for coordinates: {addressLoc[1]}, {addressLoc[0]}")
            else:
                print(f"Failed to retrieve image for coordinates: {addressLoc[1]}, {addressLoc[0]} with status code {response.status_code}")

    for image_content, city_name, index in successful_images:
        with open(os.path.join(args.output, f'{index}_{city_name}.jpg'), "wb") as file:
            file.write(image_content)

    save_processed_coords('processed_coords.csv', processed_coords)

    for i in range(len(cities_count)):
        city_count = cities_count[i]
        city_name = os.listdir(args.cities)[i]
        print(f'{city_count} images pulled from {city_name}')





if __name__ == '__main__':
    main()
