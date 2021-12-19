from torchvision import models
import torch.nn as nn

model_task1 = models.resnet18(pretrained= True)
fc_in_features = model_task1.fc.in_features
model_task1.fc = nn.Sequential(
    nn.Linear(fc_in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 20)
    #nn.Softmax(0)
)

resnet50 = models.resnet18(pretrained= True)
fc_in_features = resnet50.fc.in_features
resnet50.fc = nn.Sequential(
    nn.Linear(fc_in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 20)
    #nn.Softmax(0)
)

if __name__ == "__main__":
    print(models.resnet18())