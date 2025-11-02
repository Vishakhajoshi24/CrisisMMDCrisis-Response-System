import torch
import torch.nn as nn
from torchvision import models
from transformers import DistilBertModel

class MiniFusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.text_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.text_fc = nn.Linear(768, 128)

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.fc = nn.Linear(512, 128)
        self.image_model = resnet

        self.classifier = nn.Linear(256, 2)

    def forward(self, input_ids, attention_mask, images):
        t = self.text_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:,0,:]
        t = self.text_fc(t)
        i = self.image_model(images)
        fused = torch.cat([t, i], dim=1)
        return self.classifier(fused)
