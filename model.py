import torch
import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model


class MosFPAD(nn.Module):
    def __init__(self):
        super().__init__()
        # Load MobileNetV1 pretrained
        self.model = ptcv_get_model("mobilenet_w1", pretrained=True)

        # Remove classification head
        in_features = self.model.output.in_features
        self.model.output = nn.Identity()

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
