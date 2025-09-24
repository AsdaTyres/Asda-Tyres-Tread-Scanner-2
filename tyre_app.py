import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("tyre_cnn_demo.h5")

# Define class labels
class_labels = ["Bald", "Damaged", "Good"]

# Define color mapping for traffic-light feedback
color_map = {
    "Bald": "🟡 Bald - Consider replacing soon",
    "Damaged": "🔴 Damaged - Replace immediately",
    "Good": "🟢 Good - Tyre appears safe"
}

# Streamlit app interface
st.set_page_config(page_title="Tyre Condition Checker", layout="centered")
st.title("🛞 Tyre Condition Checker")
st.write("Upload a tyre image to check its condition using AI.")

# File uploader
uploaded_file = st.file_uploader("Upload a tyre image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Tyre Image", use_column_width=True)

    # Preprocess image
    image_resized = image.resize((128, 128))
    image_array = np.array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    predictions = model.predict(image_array)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # Display result
    st.subheader("Prediction Result")
    st.markdown(f"**Condition:** {color_map[predicted_class]}")
    st.markdown(f"**Confidence:** {confidence:.2f}%")
