import torch
import torch.nn as nn
from timm import create_model


class MosFPAD(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = create_model("tf_mobilenetv1_100", pretrained=True)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True)
        )

        self.svc = nn.Linear(256, 1)
    
    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        x = self.svc(x)
        return x.squeeze(1)

