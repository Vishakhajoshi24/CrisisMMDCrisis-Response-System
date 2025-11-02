import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as T
from transformers import DistilBertTokenizerFast

# ------------------------------
# Load Model
# ------------------------------

from model import LateFusionModel   # <-- We will add model.py next
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Load trained model
model = LateFusionModel().to(device)
model.load_state_dict(torch.load("late_fusion_crisis_model.pth", map_location=device))
model.eval()

# Image transform for ResNet
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

def predict(tweet, img):
    encoded = tokenizer(tweet, truncation=True, padding='max_length', max_length=64, return_tensors='pt').to(device)
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(encoded["input_ids"], encoded["attention_mask"], img)
        pred = torch.argmax(torch.softmax(logits, dim=1), dim=1).item()

    return "CRISIS / INFORMATIVE ✅" if pred == 1 else "NOT CRISIS 🚫"


# ------------------------------
# Streamlit UI
# ------------------------------

st.title("🚨 Crisis Detection System")
st.write("Upload an image and enter a tweet to detect crisis-related content.")

tweet = st.text_area("Enter Tweet Text Here")

uploaded_img = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if st.button("Predict"):
    if uploaded_img is None or tweet.strip() == "":
        st.error("Please upload an image AND enter a tweet.")
    else:
        img = Image.open(uploaded_img).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        result = predict(tweet, img)
        st.subheader(f"Prediction: **{result}**")
