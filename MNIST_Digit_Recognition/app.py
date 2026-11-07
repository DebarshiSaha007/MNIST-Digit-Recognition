
import streamlit as st
import numpy as np
import joblib

from PIL import Image, ImageOps
from pathlib import Path

# Load the trained model
BASE_DIR = Path(__file__).parent
model_path = BASE_DIR / "mnist_digit_model_small.pkl"
model = joblib.load(model_path)

# Page configuration
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

st.title("🔢 Handwritten Digit Recognition")

st.write(
    "Recognize handwritten digits from 0 to 9 "
    "using a machine learning model trained on MNIST."
)

# Compatibility notice
st.info(
    "⚠️ This application is designed for MNIST-style images. "
    "It works best with a single handwritten digit, centered "
    "on a plain background, with clear contrast. "
    "Images with similar characteristics may also work, "
    "but other image types may produce unreliable predictions."
)

st.divider()

# Choose image source
choice = st.radio(
    "Choose how to test the model:",
    ["Try a sample image", "Upload your own image"]
)

image = None

if choice == "Try a sample image":

    digit = st.selectbox(
        "Choose a sample handwritten digit:",
        list(range(10))
    )

    sample_path = BASE_DIR / "sample_images" / f"digit_{digit}.png"

    if sample_path.exists():
        image = Image.open(sample_path).convert("L")
    else:
        st.error("Sample image not found. Please check the sample_images folder.")

else:
    uploaded_file = st.file_uploader(
        "Upload a handwritten digit image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")

# Display and predict
if image is not None:

    st.subheader("Selected Image")
    st.image(image, caption="Input digit", width=200)

    if st.button("Predict Digit"):

        # Prepare image in MNIST format
        processed_image = ImageOps.invert(image)
        processed_image = processed_image.resize((28, 28))

        image_array = np.array(processed_image).astype("float32") / 255.0
        image_array = image_array.reshape(1, 784)

        # Make prediction
        prediction = model.predict(image_array)[0]

        st.success(f"Predicted Digit: {prediction}")
