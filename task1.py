import torch
import torch.nn as nn 
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

from dataset import video_dataset
import models

#- configs
lr = 0.007
dataset_r = 0.8             # ratio of dataset for train
momentum = 0.9
weight_decay = 1e-4
epoch = 50
save = True
batch_size_tr = 16
batch_size_val = 16

train_data_path = "./train_processed"
model_path = "./models/task1resnet18.pkl"
##

def train(tr_loader, val_loader, model, criterion, optimizer, epoch, save, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    best_acc = 0
    for i in range(epoch):
        model.train()
        losses = 0.0
        for data in tr_loader:
            image, label = data["image"], data["label"]
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            loss = criterion(model(image), label)
            # output default: reduction : mean
            loss.backward()
            optimizer.step()
            
            losses += loss.item() * image.size(0)

        acc = 0
        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            for data in val_loader:
                image, label = data["image"], data["label"]
                image = image.to(device)
                label = label.to(device)
                output = model(image)
                pred = torch.argmax(output, dim = 1)
                correct += pred.eq(label).sum()
                total += image.size(0)
            acc = correct * 1.0 / total * 100

        print("epoch%d | loss: %.03f  accuracy : %.03f%%" % (i, losses, acc))

        if save == True and best_acc < acc:
            torch.save(model.state_dict(), model_path)
        best_acc = max(best_acc, acc)

def classify_video(video_path, model):
    model.eval()
    _transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225])
    ])
    _dataset = video_dataset(video_path, "test", _transform)
    test_loader = DataLoader(_dataset, batch_size = 15, shuffle=False, num_workers=16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    result = torch.zeros(20)
    with torch.no_grad():
        for data in test_loader:
            image = data["image"]
            image = image.to(device)
            output = model(image)
            pred = torch.argmax(output, dim = 1)
            for idx in pred:
                result[idx] += 1
    _, label = result.max(0)  
    return label.item()      


if __name__ == "__main__":
    _transform = transforms.Compose([

        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225])
    ])
    _dataset = video_dataset(train_data_path, "train", _transform)
    dataset_len = len(_dataset)
    tr_dataset, val_dataset = random_split(_dataset, [dataset_len // 10, dataset_len - dataset_len // 10])
    tr_loader = DataLoader(tr_dataset, batch_size = batch_size_tr, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size = batch_size_val, shuffle=True, num_workers=16)
    
    model = models.model_task1
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay= weight_decay)

    train(tr_loader, val_loader, model, criterion=criterion, optimizer = optimizer, epoch = epoch, save=save, model_path=model_path)
