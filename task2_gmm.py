import os
from models import GMM
from dataset import audio_dataset2

train_data = "./train"
model_dir = "./models/task2_GNN.pkl"

def train(data_path, model_dir, save=True):
    model = GMM()

    for classes in os.list(data_path):
        for speeches in os.list(os.join(data_path, classes)):
            model.load(os.path.join(data_path, classes, speeches), classes)
    
    model.train()

    if save == True:
        model.save(model_dir)

def classify_audio(data_path, model_path):
    pass

if __name__ == "__main__":
    train(train_data, model_dir)
