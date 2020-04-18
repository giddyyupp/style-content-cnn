import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import model_urls
# from torchvision.models.vgg import model_urls
import config as cfg

"""
Resnet-50 model, last layer is modified to match class count.
"""


class ContentModel(nn.Module):
    def __init__(self):
        """Load the pretrained ResNet-50 and replace top fc layer."""
        super(ContentModel, self).__init__()
        model_urls['resnet50'] = model_urls['resnet50'].replace('https://', 'http://')
        self.resnet = models.resnet50(pretrained=True)

        # Newly created modules have require_grad=True by default
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, cfg.CLASS_COUNT)
        # features = list(self.resnet.children())[:-1]  # Remove last layer
        # features.extend([])  # Add our layer with N outputs
        # self.resnet = nn.Sequential(*features)  # Replace the model classifier

        print self.resnet  # display model

    def __freeze_layers__(self):
        # Freeze training for all layers
        for param in self.resnet.features.parameters():
            param.require_grad = False

    def forward(self, images):
        """Feed the image to model."""
        out = self.resnet(images)
        return out
