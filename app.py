import streamlit as st
import numpy as np
from skimage import io, transform
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the trained Random Forest model
model = joblib.load('random_forest_model.joblib')

# Define label-to-class mapping
class_mapping = {
    0: "Early Blight",
    1: "Late Blight",
    2: "Healthy"
}

st.title('Potato Disease Classifier using Random Forest')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = io.imread(uploaded_file)
    resized_image = transform.resize(image, (128, 128))

    st.image(resized_image, caption='Uploaded Image.', use_column_width=True)

    flattened_image = resized_image.reshape(1, -1)

    prediction = model.predict(flattened_image)
    prediction_probabilities = model.predict_proba(flattened_image)

    predicted_class = class_mapping.get(prediction[0], "Unknown")
    predicted_class_prob = prediction_probabilities[0][prediction[0]]
    st.write(f"Prediction: {predicted_class} with confidence: {predicted_class_prob*100}%")
