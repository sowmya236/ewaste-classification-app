
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="E-Waste Classification",
    page_icon="♻️"
)

st.title("♻️ E-Waste Classification App")
st.write("Upload an e-waste image to predict its category using AI.")

CLASS_NAMES = [
    "Battery",
    "Keyboard",
    "Microwave",
    "Mobile",
    "Mouse",
    "PCB",
    "Player",
    "Printer",
    "Television",
    "Washing Machine"
]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("ewaste_mobilenetv2.keras")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=350)

    if st.button("Classify E-Waste"):
        try:
            model = load_model()
            image = image.resize((160, 160))
            image_array = np.array(image, dtype=np.float32)
            image_array = np.expand_dims(image_array, axis=0)

            predictions = model.predict(image_array, verbose=0)[0]
            predicted_index = int(np.argmax(predictions))
            confidence = float(predictions[predicted_index]) * 100

            st.success(
                f"Predicted Category: {CLASS_NAMES[predicted_index]}"
            )
            st.write(f"Model confidence: {confidence:.2f}%")
            st.warning(
                "AI predictions may be incorrect. "
                "Please verify the result before disposal."
            )

        except Exception as error:
            st.error(f"Error loading or predicting: {error}")
