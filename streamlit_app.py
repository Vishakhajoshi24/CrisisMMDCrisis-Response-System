import streamlit as st
from PIL import Image
from inference import predict

st.set_page_config(page_title="Crisis Detector", layout="centered")

st.title("Multimodal Crisis Identification")
st.write("Upload an image and enter a tweet to check if it's a crisis situation.")

tweet = st.text_area("Enter tweet text")

uploaded_img = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_img and tweet:
    image = Image.open(uploaded_img).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    if st.button("Predict"):
        label, confidence = predict(tweet, image)

        if label == 1:
            st.success(f"Crisis Detected (Confidence: {confidence:.2f})")
        else:
            st.info(f"No Crisis Detected (Confidence: {confidence:.2f})")
