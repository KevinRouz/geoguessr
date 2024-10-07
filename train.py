import tensorflow as tf
from data_loader import StreetViewDataset
from model import create_model
from os import makedirs

# Set parameters
data_dir = 'organized_data/train'
csv_file = 'processed_coords.csv'
train_dataset = StreetViewDataset(data_dir, csv_file)

# Create and compile the model
model = create_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_dataset, epochs=10)

# Fine tuning
for layer in model.layers[0].layers[-20:]:  # Unfreeze the last 20 layers of ResNet
    layer.trainable = True

# Compile again, lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Continue training for a few more epochs to fine tune
model.fit(train_dataset, epochs=5)

# Make sure the models folder exists
makedirs('models', exist_ok=True)

# Save the model weights 
model.save_weights('models/geoguessr_v1.weights.h5')
