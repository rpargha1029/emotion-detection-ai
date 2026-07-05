import torch.nn as nn
import torchvision.models as models
from src.config import TRAIN_CONFIG


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=7, input_channels=1):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        final_size = TRAIN_CONFIG["img_size"] // 8  # after 3 poolings
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * final_size * final_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_resnet18(num_classes=7, pretrained=False, input_channels=1):
    model = models.resnet18(pretrained=pretrained)

    if input_channels != 3:
        model.conv1 = nn.Conv2d(
            input_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
