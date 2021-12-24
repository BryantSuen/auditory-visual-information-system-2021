from torch.nn.modules.activation import LogSoftmax
from torchvision import models
import torch.nn as nn

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


if __name__ == "__main__":
    print(models.resnet18())