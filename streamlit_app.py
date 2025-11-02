import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from models.early_fusion import EarlyFusionModel
from models.late_fusion import LateFusionModel
from models.hybrid_fusion import HybridFusionModel

# --------------------------------------
# ✅ App Title
# --------------------------------------
st.set_page_config(page_title="Crisis Damage Detection - Multimodal")
st.title("🔥 CrisisMMD — Multimodal Damage Classification")
st.write("Upload an image + enter text to classify disaster damage using Early, Late & Hybrid Fusion models.")

# --------------------------------------
# ✅ Load Models
# --------------------------------------
@st.cache_resource
def load_models():
    device = "cpu"
    early = EarlyFusionModel().to(device)
    late = LateFusionModel().to(device)
    hybrid = HybridFusionModel().to(device)

    # YOUR SAVED MODELS CHECKPOINTS (update paths if needed)
    try:
        early.load_state_dict(torch.load("models/early_fusion.pt", map_location="cpu"))
    except:
        pass

    try:
        late.load_state_dict(torch.load("models/late_fusion.pt", map_location="cpu"))
    except:
        pass

    try:
        hybrid.load_state_dict(torch.load("models/hybrid_fusion.pt", map_location="cpu"))
    except:
        pass

    early.eval()
    late.eval()
    hybrid.eval()
    return early, late, hybrid


early_model, late_model, hybrid_model = load_models()
device = "cpu"

# --------------------------------------
# ✅ Image Preprocessing
# --------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# --------------------------------------
# ✅ Class Labels
# --------------------------------------
idx2label = {
    0: "Don't know / Can't judge",
    1: "Little or no damage",
    2: "Mild damage",
    3: "None",
    4: "Severe damage"
}

# --------------------------------------
# ✅ Input Fields
# --------------------------------------
uploaded_img = st.file_uploader("📷 Upload Disaster Image", type=["jpg", "jpeg", "png"])
input_text = st.text_area("📝 Enter Tweet / Description Text")

fusion_choice = st.selectbox(
    "Select Fusion Model",
    ["Early Fusion", "Late Fusion", "Hybrid Fusion"]
)

if st.button("🔮 Predict Damage Level"):
    if uploaded_img is None or input_text.strip() == "":
        st.error("Please upload an image and enter text!")
        st.stop()

    # --------------------------------------
    # ✅ Preprocess Image
    # --------------------------------------
    image = Image.open(uploaded_img).convert("RGB")
    st.image(image, caption="Uploaded Image", width=250)

    img_tensor = transform(image).unsqueeze(0)

    # --------------------------------------
    # ✅ Select Model
    # --------------------------------------
    if fusion_choice == "Early Fusion":
        model = early_model
    elif fusion_choice == "Late Fusion":
        model = late_model
    else:
        model = hybrid_model

    # --------------------------------------
    # ✅ Run Prediction
    # --------------------------------------
    with torch.no_grad():
        output = model([input_text], img_tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    label = idx2label[pred]

    # --------------------------------------
    # ✅ Display Results
    # --------------------------------------
    st.subheader("Prediction Result")
    st.write(f"**Damage Level:** {label}")
    st.write(f"**Confidence:** {confidence:.2f}")

    st.success("Prediction complete!")
