import torch
import torch.nn as nn
from transformers import DistilBertModel
from torchvision import models

class LateFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Text
        self.text_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.text_fc = nn.Linear(768, 2)

        # Image
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.fc = nn.Linear(512, 2)
        self.image_model = resnet

    def forward(self, input_ids, attention_mask, image):
        text_feat = self.text_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:,0,:]
        text_logits = self.text_fc(text_feat)
        image_logits = self.image_model(image)
        
        return (text_logits + image_logits) / 2
