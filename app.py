import gdown
import os

if not os.path.exists("plant_model.keras"):
    gdown.download("https://drive.google.com/uc?id=1CdR9L38hEkOFmatwtahktbAaJoUb-J-Y", "plant_model.keras", quiet=False)

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("plant_disease_model.h5")

# Class names (IMPORTANT: same order as training)
class_names = ["early_blight", "healthy", "leaf_mold"]

# Page config
st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿")

st.title("🌿 AI Plant Disease Detection")
st.write("Upload a tomato leaf image to detect disease")

# Upload
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

def preprocess(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Analyze"):
        processed = preprocess(image)

        prediction = model.predict(processed)
        predicted_class = class_names[np.argmax(prediction)]

        st.success(f"Prediction: {predicted_class}")

        # Suggestions
        if predicted_class == "early_blight":
            st.write("💊 Use fungicide and remove infected leaves.")
        elif predicted_class == "leaf_mold":
            st.write("💊 Improve air circulation and avoid humidity.")
        else:
            st.write("✅ Plant is healthy. Maintain proper care.")