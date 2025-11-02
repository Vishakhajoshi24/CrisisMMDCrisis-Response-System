import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from transformers import DistilBertTokenizerFast

# ✅ Load our lightweight multimodal fusion model
from models.model import MiniFusion

# --------------------------------------
# ✅ App Title
# --------------------------------------
st.set_page_config(page_title="Crisis Damage Detection - Multimodal")
st.title("🔥 CrisisMMD — Multimodal Damage Classification (CPU-Friendly)")
st.write("Upload an image + enter text to classify disaster damage using a multimodal fusion model.")

# --------------------------------------
# ✅ Load Model
# --------------------------------------
@st.cache_resource
def load_model():
    device = "cpu"
    model = MiniFusion().to(device)
    model.eval()
    return model

model = load_model()
device = "cpu"

# --------------------------------------
# ✅ Tokenizer for Text
# --------------------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# --------------------------------------
# ✅ Image Preprocessing
# --------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# --------------------------------------
# ✅ Labels
# --------------------------------------
idx2label = {
    0: "Don’t know / Can't Judge",
    1: "Little or No Damage",
    2: "Mild Damage",
    3: "Moderate Damage",
    4: "Severe Damage"
}

# --------------------------------------
# ✅ Prediction Function
# --------------------------------------
def predict_damage(text, image):
    encoded = tokenizer(
        text, truncation=True, padding="max_length",
        max_length=64, return_tensors="pt"
    )

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(
            encoded["input_ids"],
            encoded["attention_mask"],
            img_tensor
        )
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax().item()
        confidence = probs[0][pred].item()

    return idx2label[pred], confidence

# --------------------------------------
# ✅ User Inputs
# --------------------------------------
uploaded_img = st.file_uploader("📷 Upload Disaster Image", type=["jpg", "jpeg", "png"])
input_text = st.text_area("📝 Enter Tweet / Description Text")

if st.button("🔮 Predict Damage Level"):
    if uploaded_img is None or input_text.strip() == "":
        st.error("Please upload an image AND enter text.")
        st.stop()

    image = Image.open(uploaded_img).convert("RGB")
    st.image(image, caption="Uploaded Image", width=250)

    # ✅ Run prediction
    label, confidence = predict_damage(input_text, image)

    # ✅ Output
    st.subheader("✅ Prediction Result")
    st.write(f"**Damage Level:** {label}")
    st.write(f"**Confidence:** {confidence:.2f}")

    st.success("Prediction complete!")
