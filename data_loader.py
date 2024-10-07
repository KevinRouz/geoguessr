import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf

class StreetViewDataset(tf.keras.utils.Sequence):
    def __init__(self, data_dir, csv_file, batch_size=32, img_size=(224, 224), **kwargs):
        super().__init__(**kwargs)  # Call the parent class's constructor (throws warning unless included)

        self.data_dir = data_dir
        self.csv_data = pd.read_csv(csv_file, header=None)
        self.batch_size = batch_size
        self.img_size = img_size

        # Prepare data
        self.image_paths = []
        self.labels = []
        self.city_to_index = {'austin': 0, 'dallas': 1, 'elpaso': 2, 'houston': 3, 'sanantonio': 4}  # Map for cities to index
        
        for city_folder in os.listdir(data_dir):
            city_path = os.path.join(data_dir, city_folder)
            if os.path.isdir(city_path):
                for img_file in os.listdir(city_path):
                    if img_file.endswith('.jpg'):
                        index = int(img_file.split('_')[0]) - 1
                        lat, long, city_name = self.csv_data.iloc[index].values
                        self.image_paths.append(os.path.join(city_path, img_file))
                        self.labels.append((lat, long, self.city_to_index[city_folder]))  # Store lat, long, and city index
        
        self.labels = np.array(self.labels)
    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_x = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_y = self.labels[index * self.batch_size:(index + 1) * self.batch_size][:, 2]  # Get only city indices

        images = []
        for img_path in batch_x:
            img = cv2.imread(img_path)
            img = cv2.resize(img, self.img_size)
            images.append(img)

        return np.array(images), np.array(batch_y, dtype=np.int32)