from torchvision import models, transforms
from torchvision.models import resnet
import torch.nn as nn
import torch

from dataset import video_dataset
from utils import read_video

resnet = models.resnet101(pretrained=True)

resnet.fc = nn.Linear(in_features=2048, out_features= 10, bias = True)
print(resnet.fc)
video, video_fps = read_video("./train/ID1/014.mp4")

print(video.shape)
print(len(video))

for idx in video:
    print("ok")
preprocess = transforms.Compose([
    transforms.ToTensor()
    ]
)


# data = torch.stack([preprocess(video[0]), preprocess(video[1])], dim=0)
# print("data size: ")
# resnet.eval()
# out = resnet(data)

# print(out.size())
# print(out)

# criterion = nn.CrossEntropyLoss()

# loss = criterion(out, out)
# print("okkkkk")
# print(loss)
# a = "asasa.mp4"
# b = a + str(10)