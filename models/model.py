import torch
import torch.nn as nn
from transformers import DistilBertModel
from torchvision import models

class MiniFusion(nn.Module):
    def __init__(self):
        super().__init__()

        # Text Encoder (DistilBERT)
        self.text_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.text_fc = nn.Linear(768, 128)

        # Image Encoder (ResNet18)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.fc = nn.Linear(512, 128)
        self.image_model = resnet

        # Fusion Classifier
        self.classifier = nn.Linear(256, 5)  # 5 damage categories

    def forward(self, input_ids, attention_mask, images):
        text_features = self.text_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]
        text_features = self.text_fc(text_features)

        image_features = self.image_model(images)

        fused = torch.cat([text_features, image_features], dim=1)
        return self.classifier(fused)
