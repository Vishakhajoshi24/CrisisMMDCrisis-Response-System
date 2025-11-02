import torch
from PIL import Image
from transformers import DistilBertTokenizerFast
import torchvision.transforms as T
from model_definitions import LateFusionModel

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# Load model
model = LateFusionModel().to(device)
model.load_state_dict(torch.load("model/late_fusion_crisis_model.pth", map_location=device))
model.eval()

def predict(tweet, img):
    # Encode text
    enc = tokenizer(tweet, return_tensors="pt", padding="max_length",
                    truncation=True, max_length=64).to(device)
    
    # Process image
    image_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(enc["input_ids"], enc["attention_mask"], image_tensor)
        prob = torch.softmax(logits, dim=1)
        pred = torch.argmax(prob).item()

    return pred, prob[0][pred].item()
