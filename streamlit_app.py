import streamlit as st
from app.predict import Predictor
import os

# Set up the predictor
model_path = "app/breast_cancer_vit.pth"  # Path to your model weights
predictor = Predictor(model_path=model_path)

# Streamlit App
st.title("Breast Cancer Classification with Vision Transformer")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    with st.spinner("Classifying..."):
        image_path = os.path.join("dataset", uploaded_file.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.read())

        prediction = predictor.predict(image_path)
        st.success("Classification Complete!")
        st.write(f"**Benign Probability:** {prediction['Benign']:.2f}")
        st.write(f"**Malignant Probability:** {prediction['Malignant']:.2f}")
