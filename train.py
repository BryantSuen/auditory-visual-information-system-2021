import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.serialization import load
from torchvision import transforms, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import video_dataset

def train(tr_loader, val_loader, model, criterion, optimizer, epoch, device):
    for i in tqdm(range(epoch)):
        model.train()
        if device == "cuda":
            model.to(torch.device("cuda"))
        loss = 0.0
        cnt = 0
        for data in tr_loader:
            image, label = data["image"], data["label"]
            if device == "cuda":
                image = torch.autograd.Variable(image.cuda())
                label = torch.autograd.Variable(label.cuda())
            optimizer.zero_grad()

            output = criterion(model(image), label)
            # output default: reduction : mean
            output.backward()
            optimizer.step()
            loss += output.item()
            cnt += 1
            if cnt % 20 == 0:
                print("%d: loss = %f" %(cnt, loss))

        acc = validate(model, val_loader)
        print("epoch%d | loss: %.03f  accuracy : %.03f%%" % (i, loss, acc))

def validate(model, loader):
    model.eval()
    correct = 0
    total = 0
    for data in loader:
        image, label = data["image"], data["label"]
        image = image.cuda()
        label = label.cuda()
        # video, label = torch.autograde.Variable()
        output = model(image)
        pred = torch.argmax(output)
        correct += (pred == label).sum().float()
        total += len(label)
    acc = 100 * correct * 1.0 / total
    return acc

if __name__ == "__main__":
    _transform = transforms.Compose([

        transforms.ToTensor()
    ])
    _dataset = video_dataset("./train_processed", "train", _transform)
    dataset_r = 0.8
    dataset_len = len(_dataset)
    tr_dataset, val_dataset = random_split(_dataset, [dataset_len // 10, dataset_len - dataset_len // 10])
    tr_loader = DataLoader(tr_dataset, batch_size = 32, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size = 16, shuffle=False, num_workers=16)
    
    model = models.resnet18(pretrained= True)
    fc_in_features = model.fc.in_features
    model.fc = nn.Linear(in_features= fc_in_features, out_features= 20)
    criterion = nn.CrossEntropyLoss()
    lr = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum= 0.9, weight_decay=1e-4)

    train(tr_loader, val_loader, model, criterion=criterion, optimizer = optimizer, epoch = 20, device="cuda")

