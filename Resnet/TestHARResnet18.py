import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import numpy as np
from utils.Helpers import HARDataloader
from torch.utils.data import DataLoader
import os

from PIL import Image
import matplotlib.pyplot as plt

from torchvision.models import ResNet18_Weights

DatasetPath = "./train/"
AnnotationFile = "./Training_set.csv"
HAR_data = HARDataloader(ann_file=AnnotationFile, rootPath=DatasetPath)


train_dataloader = DataLoader(HAR_data, batch_size=64, shuffle=True)

num_epochs = 10


class ConvNet(nn.Module):
    def __init__(self, numberofclasses):
        super(ConvNet, self).__init__()

        # Choose a pre-trained model
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # freeze the base model.
        for param in self.model.parameters():
            param.requires_grad = False

        # Modify the model for your classification task
        in_features = self.model.fc.in_features
        
        # Add more dense layers
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 1000),  # Add a dense layer with 512 output features
            nn.BatchNorm1d(1000),
            nn.ReLU(),                 # Add a ReLU activation function
            nn.Dropout(0.5),           # Add dropout for regularization
            nn.Linear(1000, 512),  # Modify the last layer for the number of classes
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),           # Add dropout for regularization
            nn.Linear(512, numberofclasses)  # Modify the last layer for the number of classes
        )
        
        
        #self.model.fc = nn.Linear(in_features, numberofclasses)

        # Additional FC Layers for some learning.
        self.fc2 = nn.Linear(1000,512)
        self.fc3 = nn.Linear(512,128)
        self.fc4 = nn.Linear(128,64)
        self.fc5 = nn.Linear(64,numberofclasses)

        self.Drop4 = nn.Dropout(p=0.4)
        self.Drop5 = nn.Dropout(p=0.5)

        self.Softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.model(x)
        #x = self.fc2(x)
        #x = self.Drop5(x)
        #x = self.fc3(x)
        #x = self.Drop5(x)
        #x = self.fc4(x)
        #x = self.Drop5(x)
        #x = self.fc5(x)
        #x = self.Softmax(x)
        return x

model = ConvNet(numberofclasses=len(HAR_data.classes))

# Check if cuda is available
use_cuda = torch.cuda.is_available()

# Set proper device based on cuda availability 
device = torch.device("cuda" if use_cuda else "cpu")
print("Torch device selected: ", device)


model.load_state_dict(torch.load("./models/HAR_resnet18_final_2023_11_13__18255_c43374_50.pth"))
model.to(device=device)
model.eval()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

transform = transforms.Compose([ 
        transforms.Resize((224,224), antialias=True),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]) 

import glob
testImages = glob.glob("./test/*.jpg")

cnt = 0
for imgPath in testImages:
    with torch.no_grad():
        PilImage = Image.open(f"{testImages[cnt]}")
        input = transform(PilImage)
        

        logits = model(input.float().unsqueeze(0).to(device))

        maxindex = logits.argmax(dim=-1)

        predClass = HAR_data.classes[maxindex]

        classDir = f"output/{predClass}"
        imageSavePath = f"{classDir}/{cnt}.jpg"
        os.makedirs(classDir, exist_ok=True)

        PilImage.save(imageSavePath)
        print(f"Predicted: {predClass} conf: {logits[0][maxindex]}")

        cnt+=1

        # if cnt>500:
        #     break