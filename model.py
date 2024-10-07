import tensorflow as tf
from tensorflow.keras.applications import ResNet50

def create_model(num_classes=5):
    # Load the pre-trained ResNet50 model, excluding the top layers
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Create a new model on top of the base model
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),  # Global Average Pooling layer
        tf.keras.layers.Dense(128, activation='relu'),  # Hidden layer
        tf.keras.layers.Dropout(0.5),  # Dropout layer for regularization
        tf.keras.layers.Dense(num_classes, activation='softmax')  # Output layer
    ])
    
    return model