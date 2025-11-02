import torch
import torch.nn as nn
from transformers import DistilBertModel
from torchvision import models

class LateFusionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # TEXT ENCODER (DistilBERT)
        self.text_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.text_fc = nn.Linear(768, 128)

        # IMAGE ENCODER (ResNet18)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.fc = nn.Linear(512, 128)
        self.image_model = resnet

        # BINARY CLASSIFIER
        self.classifier = nn.Linear(256, 2)   # Crisis / Not Crisis

    def forward(self, input_ids, attention_mask, images):
        text_feat = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]
        text_feat = self.text_fc(text_feat)

        img_feat = self.image_model(images)

        fused = torch.cat([text_feat, img_feat], dim=1)

        return self.classifier(fused)
