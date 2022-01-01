from torch.nn.modules.activation import LogSoftmax
from torch.nn.modules.conv import Conv2d
from torchvision import models
import torch.nn as nn
import torch

model_task1 = models.resnet34(pretrained= True)
fc_in_features = model_task1.fc.in_features
model_task1.fc = nn.Sequential(
    nn.Linear(fc_in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.25),
    nn.Linear(256, 20),
    nn.LogSoftmax(dim=1)
    #nn.Softmax(0)  # crossentropy already has a softmax
)

model_task2 = models.resnet34(pretrained=True)
model_task2.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
fc_in_features_2 = model_task2.fc.in_features
model_task2.fc = nn.Sequential(
    nn.Linear(fc_in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.25),
    nn.Linear(256, 20),
    nn.LogSoftmax(dim=1)
)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2),
            # nn.Conv2d(8, 16, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(384, 256, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            # #nn.Dropout(),
            # nn.Linear(256 * 6 * 6, 4096),
            # nn.ReLU(inplace=True),
            # #nn.Dropout(),
            # nn.Linear(4096, 4096),
            # nn.ReLU(inplace=True),
            nn.Linear(8704, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(256, 20),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # print("shape: ", x.shape)
        # print("shape", x.shape)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    print(models.resnet18())