import numpy as np
import cv2
import tensorflow as tf
from model import create_model

def predict_city_and_coords(model, image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    predictions = model.predict(img)
    
    city_index = np.argmax(predictions)  # Get the index of the city with highest confidence
    confidence = np.max(predictions)  # Get the confidence score
    
    city_name = ['austin', 'dallas', 'elpaso', 'houston', 'sanantonio'][city_index]
    
    return city_name, confidence

if __name__ == "__main__":
    model = create_model()
    model.load_weights('geoguessr_v1.weights.h5')  # Load the model
    city_name, confidence = predict_city_and_coords(model, 'path_to_test_image.jpg')
    print(f"Predicted city: {city_name}, Confidence: {confidence}")
