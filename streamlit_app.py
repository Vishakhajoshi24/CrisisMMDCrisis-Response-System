import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from transformers import DistilBertTokenizerFast

from model import LateFusionModel

# -----------------------------------------------------
# ✅ Streamlit Page Config
# -----------------------------------------------------
st.set_page_config(page_title="Crisis Detection System")
st.title("🚨 Crisis Detection (Image + Tweet)")
st.write("Upload an image and enter tweet text to detect if it's crisis-related.")

# -----------------------------------------------------
# ✅ Load Model + Tokenizer
# -----------------------------------------------------
@st.cache_resource
def load_model():
    model = LateFusionModel().to("cpu")
    model.load_state_dict(torch.load("late_fusion_crisis_model.pth", map_location="cpu"))
    model.eval()
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    return model, tokenizer

model, tokenizer = load_model()

# -----------------------------------------------------
# ✅ Image Transform
# -----------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# -----------------------------------------------------
# ✅ Predict Function
# -----------------------------------------------------
def predict_crisis(text, image):
    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=64,
        return_tensors="pt"
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

    return pred, probs[0][pred].item()

# -----------------------------------------------------
# ✅ UI Inputs
# -----------------------------------------------------
tweet = st.text_area("📝 Enter Tweet Text")
uploaded_img = st.file_uploader("📷 Upload Image", type=["jpg", "jpeg", "png"])

if st.button("🔮 Predict"):
    if uploaded_img is None or tweet.strip() == "":
        st.error("Please upload an image and enter text!")
        st.stop()

    img = Image.open(uploaded_img).convert("RGB")
    st.image(img, caption="Uploaded Image", width=250)

    pred, confidence = predict_crisis(tweet, img)

    label = "✅ Crisis / Informative" if pred == 1 else "❌ Not Crisis"

    st.subheader("Prediction Result")
    st.write(f"**Category:** {label}")
    st.write(f"**Confidence:** {confidence:.2f}")

    st.success("Prediction complete!")
