import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.optim import optimizer
from torch.utils.data import DataLoader, random_split
from torchaudio import transforms

from dataset import audio_dataset
import models
from MFCC import MFCC
from utils import read_audio

#- configs
lr = 0.01

#- configs  #70%
#lr = 0.007
#dataset_r = 0.8             # ratio of dataset for train
#momentum = 0.9
#weight_decay = 1e-4
#epoch = 50
#save = True
#batch_size_tr = 16
#batch_size_val = 16

#new configs
#lr = 0.0075

dataset_r = 0.8             # ratio of dataset for train
momentum = 0.9
weight_decay = 5e-4
epoch = 20
save = True
batch_size_tr = 32
batch_size_val = 16

train_data_path = "./train_audio/"
model_path = "./models/task2_CNN.pkl"
##

def train(tr_loader, val_loader, model, criterion, optimizer, epoch, save, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)
    best_acc = 0
    for i in range(epoch):
        for param_group in optimizer.param_groups:
            if i in [30, 60]:
                param_group['lr'] *= 0.1
        model.train()
        losses = 0.0
        for data in tr_loader:
            speech, label = data["speech"], data["label"]
            speech = speech.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            loss = criterion(model(speech), label)
            # output default: reduction : mean
            loss.backward()
            optimizer.step()
            
            losses += loss.item()

        acc = 0
        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            for data in val_loader:
                speech, label = data["speech"], data["label"]
                speech = speech.to(device)
                label = label.to(device)
                output = model(speech)
                pred = torch.argmax(output, dim = 1)
                #print("pred: ", pred)
                #print("label: ", label)
                correct += pred.eq(label).sum().item()
                total += speech.size(0)
            acc = correct * 1.0 / total * 100

        print("epoch%d | loss: %.03f  accuracy : %.03f%%" % (i, losses, acc))

        if save == True and best_acc < acc:
            torch.save(model.state_dict(), model_path)
        best_acc = max(best_acc, acc)

def classify_audio(audio_path, model):
    model.eval()
    result = torch.zeros(20)
    _dataset = audio_dataset(audio_path, mode="test")
    test_loader = DataLoader(_dataset, batch_size = 15, shuffle=False, num_workers=16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    with torch.no_grad():
        for data in test_loader:
            speech = data["speech"]
            speech = speech.to(device)
            output = model(speech)
            pred = torch.argmax(output, dim = 1)
            for idx in pred:
                result[idx] += 1
    _, label = result.max(0)  
    return label.item()          


if __name__ == "__main__":
    # _transform = transforms.MFCC()
    _dataset = audio_dataset(train_data_path, "train")
    dataset_len = len(_dataset)
    tr_dataset, val_dataset = random_split(_dataset, [int(dataset_len * dataset_r), dataset_len - int(dataset_len * dataset_r)])
    tr_loader = DataLoader(tr_dataset, batch_size = batch_size_tr, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size = batch_size_val, shuffle=True, num_workers=8)
    
    model = models.CNN()
    # model = models.model_task2
    criterion = nn.NLLLoss()
    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay= weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay= weight_decay)
    train(tr_loader, val_loader, model, criterion=criterion, optimizer = optimizer, epoch = epoch, save=save, model_path=model_path)
