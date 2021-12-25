import torch
import os

from torchvision import transforms
from utils import read_video
from PIL import Image
from image_transforms import image_transforms
import matplotlib.pyplot as plt

labels = {
    "ID1": 0,
    "ID2": 1,
    "ID3": 2,
    "ID4": 3,
    "ID5": 4,
    "ID6": 5,
    "ID7": 6,
    "ID8": 7,
    "ID9": 8,
    "ID10":9 ,
    "ID11": 10,
    "ID12": 11,
    "ID13": 12,
    "ID14": 13,
    "ID15": 14,
    "ID16": 15,
    "ID17": 16,
    "ID18": 17,
    "ID19": 18,
    "ID20": 19
}

class audio_dataset(torch.utils.data.Dataset):
    def __init__(self, path = "./train", mode = "train", transform = None):

        self.mode = mode
        self.images = []
        self.ids = []
        self.transform = transform
        self.path = path
        self.length = 0

        if self.mode == "train":
            image_classes = os.listdir(self.path)
            for classes in image_classes:
                image_dir = os.listdir(os.path.join(self.path, classes))
                self.images += list(map(lambda x: classes + "/" + x, image_dir))
                self.ids += [classes] * len(image_dir)

        else:
            video, video_fps = read_video(self.path)
            self.length = len(video)

    def __len__(self):
        if self.mode == "train":
            return len(self.images)
        else:
            return self.length

    def __getitem__(self, idx):
        if self.mode == "train":
            image = Image.open(os.path.join(self.path, self.images[idx]))
            if self.transform is not None:
                image = self.transform(image)
            label = labels[self.ids[idx]]
            return {"image": image, "label": label}
        else:
            video, video_fps = read_video(self.path)
            image = video[idx].copy()
            image = Image.fromarray(image)
            if self.transform is not None:
                image = self.transform(image)
            return {"image": image}