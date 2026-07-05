import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model("model_new.h5")
st.title("MNIST Digit Classifier")

uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L").resize((28, 28))
    img_array = np.array(image).reshape(1, 28, 28) / 255.0

    if img_array.mean() > 0.5:
        img_array = 1 - img_array

    prediction = model.predict(img_array)
    confidence = np.max(prediction) * 100
    predicted_digit = np.argmax(prediction)

    st.image(image, caption="Processed Image (28x28 grayscale)", width=150)
    st.subheader(f"Predicted digit: {predicted_digit}")
    st.write(f"Confidence: {confidence:.1f}%")

    if confidence < 50:
        st.warning("Low confidence — this may not be a clear handwritten digit.")
else:
    st.info("Please upload an image of a handwritten digit (0–9) to get a prediction.")
