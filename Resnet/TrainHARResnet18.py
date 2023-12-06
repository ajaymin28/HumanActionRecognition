import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import numpy as np
from utils.Helpers import HARDataloader
from torch.utils.data import DataLoader
import os
from torchvision.models import ResNet18_Weights
import argparse
from torch.optim import lr_scheduler
import uuid
from tqdm import tqdm
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg') 


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
            nn.Linear(1000, numberofclasses),  # Modify the last layer for the number of classes
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
        return x
    

def validate_model(model, val_dataloader, criterion,device):
    # Evaluation on validation set
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in val_dataloader:

            labels = labels.to(device)

            outputs = model(inputs.float().to(device))
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            maxindexes = outputs.argmax(dim=-1)
            GT_maxindexes = labels.argmax(dim=-1)

            batch_correct = (maxindexes == GT_maxindexes).sum().item()
            correct += batch_correct

            # _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # correct += (predicted == labels.to(device)).sum().item()

    val_loss = round(float(total_loss/total),4)
    val_acc = round((100 * correct) / total, 4)

    # print(f'Validation loss: {val_loss} Accuracy: {val_acc}%')  

    return val_loss, val_acc

def train_model(FLAGS, model, train_dataloader, val_dataloader, optimizer, scheduler, criterion, device, log_dir="logs"):

    os.makedirs(log_dir,exist_ok=True)

    best_model_params_path = os.path.join(log_dir, 'best_model_params.pt')
    logs_path = os.path.join(log_dir, 'logs.txt')

    logs_file = open(logs_path, "w")

    Accuracies = []
    Losses = []
    Val_Accuracies = []
    Val_losses = []

    BestAcc = 0

    model.to(device=device)
    print("Torch device selected: ", device)
    logs_file.write(f"Torch device selected:{device} \n")

    # Training loop
    for epoch in tqdm(range(FLAGS.num_epochs)):

        total = 0
        correct = 0

        total_train_loss = 0

        model.train(mode=True)

        for inputs, labels in train_dataloader:
            optimizer.zero_grad()

            labels = labels.to(device)

            outputs = model(inputs.float().to(device))
            loss = criterion(outputs, labels)

            maxindexes = outputs.argmax(dim=-1)
            GT_maxindexes = labels.argmax(dim=-1)

            total += labels.size(0)

            batch_correct = (maxindexes == GT_maxindexes).sum().item()
            correct += batch_correct

            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()

        scheduler.step()

        train_loss = round(float(total_train_loss/total),4)
        train_acc = round((100 * correct) / total, 4)

        Accuracies.append(train_acc)
        Losses.append(train_loss)

        # validation acc and loss calculation
        val_loss, val_acc = validate_model(model, val_dataloader, criterion, device=device)
        Val_Accuracies.append(val_acc)
        Val_losses.append(val_loss)

        # print("after validating model")
        
        saveModel = False
        minLoss = None
        if len(Losses)>1:
            minLoss = min(Losses)
            if Losses[-1]<=minLoss:
                saveModel = True
        else:
            saveModel = True

        if saveModel:
            BestAcc = val_acc
            torch.save(model.state_dict(), best_model_params_path)
            logs_file.write(f"Saving model for best loss: {minLoss} val loss: {val_loss} val acc:{BestAcc} acc: {train_acc}  loss:{train_loss} \n")


        print(f"Epoch {epoch+1} Loss: {train_loss} Acc: {train_acc}  val_loss: {val_loss}  val_acc: {val_acc}")
        logs_file.write(f"Epoch {epoch+1} Loss: {train_loss} Acc: {train_acc}  val_loss: {val_loss}  val_acc: {val_acc}\n")

        os.makedirs(f"{log_dir}", exist_ok=True)

        try:
            # Plot the loss
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, len(Losses)+1), Losses, marker='o', linestyle='-', color='b')
            plt.plot(range(1, len(Val_losses)+1), Val_losses, marker='o', linestyle='-', color='g')
            plt.title(f'Training & Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.grid(True)
            plt.savefig(f"{log_dir}/{unique_run_id}_loss.png")
            # plt.show()
            
            plt.close()
            

            # Plot the accuracy
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, len(Accuracies)+1), Accuracies, marker='o', linestyle='-', color='r')
            plt.plot(range(1, len(Val_Accuracies)+1), Val_Accuracies, marker='o', linestyle='-', color='y')
            plt.title(f'Training & validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy (%)')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.grid(True)
            plt.savefig(f"{log_dir}/{unique_run_id}_acc.png")
            # plt.show()
            
            plt.close()
            
        except Exception as e:
            print(f"Error plotting acc and loss E: {e}")


    logs_file.close()

    return Losses, Accuracies, Val_losses, Val_Accuracies, model

if __name__=="__main__":

    parser = argparse.ArgumentParser('Human Action Recognition Training Script')

    parser.add_argument('--learning_rate',
                        type=float, default=0.00001,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=5,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=4,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs',
                        help='Directory to put logging.')
    parser.add_argument('--random_seed',
                        type=int,
                        default=43,
                        help='Random seed to shuffle and split the dataset')
    
    parser.add_argument('--dataset_path',
                        type=str,
                        default="./train",
                        help='Dataset images path')
    
    parser.add_argument('--annotation_file_path',
                        type=str,
                        default="./Training_set.csv",
                        help='Dataset annotations file path')
    
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()


    # DatasetPath = "/lustre/fs1/home/cap6411.student10/computervision/project/data/Human Action Recognition/train/"
    # AnnotationFile = "/lustre/fs1/home/cap6411.student10/computervision/project/data/Human Action Recognition/Training_set.csv"
    DatasetPath = FLAGS.dataset_path # "./train"
    AnnotationFile = FLAGS.annotation_file_path #"./Training_set.csv"
    
    HAR_data = HARDataloader(ann_file=AnnotationFile, rootPath=DatasetPath)

    generator1 = torch.Generator().manual_seed(FLAGS.random_seed)
    # HAR_train_ds, HAR_val_ds, HAR_test_ds = torch.utils.data.random_split(HAR_data, [0.7, 0.2, 0.1], generator=generator1)
    HAR_train_ds, HAR_val_ds = torch.utils.data.random_split(HAR_data, [0.8, 0.2], generator=generator1)

    train_dataloader = DataLoader(HAR_train_ds, batch_size=FLAGS.batch_size, shuffle=True)
    val_dataloader = DataLoader(HAR_val_ds, batch_size=FLAGS.batch_size, shuffle=True)
    # test_dataloader = DataLoader(HAR_test_ds, batch_size=FLAGS.batch_size, shuffle=False)

    model = ConvNet(numberofclasses=len(HAR_data.classes))

    # Check if cuda is available
    use_cuda = torch.cuda.is_available()

    # Set proper device based on cuda availability 
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model.parameters(), lr=FLAGS.learning_rate, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    unique_run_id = str(uuid.uuid4())[0:6]
    datetimenow = datetime.now()
    datetime_fmt = f"{datetimenow.year}_{datetimenow.month}_{datetimenow.day}__{datetimenow.hour}{datetimenow.minute}{datetimenow.second}"

    outputDir = f"logs/{datetime_fmt}/{unique_run_id}"
    Losses, Accuracies, Val_losses, Val_Accuracies, model = train_model(FLAGS=FLAGS, 
                model=model, 
                train_dataloader=train_dataloader, 
                val_dataloader=val_dataloader, 
                optimizer=optimizer_ft, 
                scheduler=exp_lr_scheduler,criterion=criterion,
                device=device,
                log_dir=outputDir
                )
    
    os.makedirs(f"{outputDir}", exist_ok=True)

    try:
        # Plot the loss
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, FLAGS.num_epochs + 1), Losses, marker='o', linestyle='-', color='b')
        plt.plot(range(1, FLAGS.num_epochs + 1), Val_losses, marker='o', linestyle='-', color='g')
        plt.title(f'Training & Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.grid(True)
        plt.savefig(f"{outputDir}/{unique_run_id}_loss.png")
        # plt.show()
        plt.close()
        

        # Plot the accuracy
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, FLAGS.num_epochs + 1), Accuracies, marker='o', linestyle='-', color='r')
        plt.plot(range(1, FLAGS.num_epochs + 1), Val_Accuracies, marker='o', linestyle='-', color='y')
        plt.title(f'Training & validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.grid(True)
        plt.savefig(f"{outputDir}/{unique_run_id}_acc.png")
        # plt.show()
        plt.close()
        

    except Exception as e:
        print(f"Error plotting acc and loss E: {e}")

    torch.save(model.state_dict(), f'{outputDir}/HAR_resnet18_final_{datetime_fmt}_{unique_run_id}_{FLAGS.num_epochs}.pth')