import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class UNetWithPretrainedEncoder(nn.Module):
    def __init__(self, encoder_name='resnet34', in_channels=3, num_classes=1, activation=None):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
            activation=activation
        )

    def forward(self, x):
        return self.model(x)
    
