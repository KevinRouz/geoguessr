import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt

# print(tf.__version__)
# print(tf.test.is_built_with_cuda())
# print(tf.config.list_physical_devices('GPU'))

# Path to the dataset
train_dir = "organized_data/train"
val_dir = "organized_data/validation"
test_dir = "organized_data/test"

# Image parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='categorical')

val_generator = val_datagen.flow_from_directory(val_dir,
                                                target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                batch_size=BATCH_SIZE,
                                                class_mode='categorical')

test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                  batch_size=BATCH_SIZE,
                                                  class_mode='categorical')

# Define the CNN model
def create_model():
    base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                                                   include_top=False,
                                                   weights='imagenet')

    base_model.trainable = False  # Freeze the base model

    model = models.Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    
    # City Classification Head
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    city_output = layers.Dense(5, activation='softmax', name='city_output')(model.output)

    # Latitude and Longitude Regression Head
    lat_long_head = layers.Dense(512, activation='relu')(model.output)
    lat_long_head = layers.Dense(256, activation='relu')(lat_long_head)
    lat_long_output = layers.Dense(2, name='lat_long_output')(lat_long_head)

    # Define the final model
    model = models.Model(inputs=base_model.input, outputs=[city_output, lat_long_output])

    return model

# Compile the model
model = create_model()

model.compile(optimizer='adam',
              loss={'city_output': 'categorical_crossentropy', 'lat_long_output': 'mse'},
              metrics={'city_output': 'accuracy', 'lat_long_output': 'mae'})

# Train the model
history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // BATCH_SIZE,
                    validation_data=val_generator,
                    validation_steps=val_generator.samples // BATCH_SIZE,
                    epochs=10)

# Save the model
model.save("street_view_model.h5")

# Plot training history
plt.plot(history.history['city_output_accuracy'], label='city accuracy')
plt.plot(history.history['val_city_output_accuracy'], label='val city accuracy')
plt.plot(history.history['lat_long_output_mae'], label='lat/long MAE')
plt.legend()
plt.show()

