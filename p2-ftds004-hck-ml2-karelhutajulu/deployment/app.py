import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the model
with open('cnn_model.pkl', 'rb') as f:
    cnn_model = pickle.load(f)

# Define the class labels
class_names = ['nofire', 'fire']

# Define the input shape for the model
img_width, img_height = 256, 256

# Define the function for classifying the image
def classify_image(image_data):
    # Convert the image data to a numpy array
    img = Image.open(image_data)
    img = img.resize((img_width, img_height))
    img = np.array(img)
    img = img / 255.0
    img = img.reshape(1, img_width, img_height, 3)

    # Use the model to predict the class label
    pred = cnn_model.predict(img)[0]
    pred_class = np.argmax(pred)
    pred_label = class_names[pred_class]
    pred_prob = round(pred[pred_class], 4)

    return pred_label, pred_prob

# Set up the web application using Streamlit
st.title('Image Classification with CNN')
st.write('This app classifies images as either no fire or on fire using a CNN model.')
image_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

# If an image is uploaded, classify it and display the result
if image_file is not None:
    # Display the uploaded image
    image_data = image_file.read()
    st.image(image_data, caption='Uploaded Image', use_column_width=True)

    # Classify the image and display the result
    pred_label, pred_prob = classify_image(image_file)
    st.write(f'Prediction: {pred_label} (Probability: {pred_prob})')
