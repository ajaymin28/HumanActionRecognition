import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import numpy as np
from utils.Helpers import CLIPDataset, convert_models_to_fp32
from torch.utils.data import DataLoader
import os
import argparse
from torch.optim import lr_scheduler
import uuid
from tqdm import tqdm
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib

import clip
from transformers import CLIPProcessor, CLIPModel

matplotlib.use('Agg') 

def validate_model(model, val_dataloader,loss_img, loss_txt, device):
    # Evaluation on validation set
    model.train(mode=False)
    total_loss = 0

    Batch_losses = []

    with torch.no_grad():
        pbar = tqdm(val_dataloader, total=len(val_dataloader))
        for batch in pbar:
            if batch is None:
                continue

            images,texts = batch

            images= images.to(device)
            texts = texts.to(device)

            # Forward pass
            logits_per_image, logits_per_text = model(images, texts)

            # Compute loss
            ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
            total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2

            Batch_losses.append(total_loss.item())

            pbar.set_description(f"Validation Loss: {total_loss.item():.4f}")



    val_loss = np.array(Batch_losses)
    val_loss = val_loss.mean()

    # print(f'Validation loss: {val_loss} Accuracy: {val_acc}%')  

    return val_loss


if __name__=="__main__":

    parser = argparse.ArgumentParser('Human Action Recognition Training Script')

    parser.add_argument('--learning_rate',
                        type=float, default=0.00001,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=50,
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

    # Check if cuda is available
    use_cuda = torch.cuda.is_available()

    # Set proper device based on cuda availability 
    device = torch.device("cuda" if use_cuda else "cpu")


    # Load the CLIP model and processor
    # model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    # processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load pre-trained CLIP model
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)


    #DatasetPath = "/lustre/fs1/home/cap6411.student10/computervision/project/data/Human Action Recognition/train/"
    #AnnotationFile = "/lustre/fs1/home/cap6411.student10/computervision/project/data/Human Action Recognition/Training_set.csv"
    DatasetPath = FLAGS.dataset_path # "./train"
    AnnotationFile = FLAGS.annotation_file_path #"./Training_set.csv"
    
    
    HAR_data = CLIPDataset(ann_file=AnnotationFile,img_preprocessing_fn=preprocess,rootPath=DatasetPath)

    generator1 = torch.Generator().manual_seed(FLAGS.random_seed)
    # HAR_train_ds, HAR_val_ds, HAR_test_ds = torch.utils.data.random_split(HAR_data, [0.7, 0.2, 0.1], generator=generator1)
    HAR_train_ds, HAR_val_ds = torch.utils.data.random_split(HAR_data, [0.8, 0.2], generator=generator1)

    train_dataloader = DataLoader(HAR_train_ds, batch_size=FLAGS.batch_size, shuffle=True)
    val_dataloader = DataLoader(HAR_val_ds, batch_size=FLAGS.batch_size, shuffle=True)
    # test_dataloader = DataLoader(HAR_test_ds, batch_size=FLAGS.batch_size, shuffle=False)
    
    
    for parm in model.parameters():
        parm.requires_grad = False
    model.visual.proj.requires_grad = True
    model.text_projection.requires_grad = True


    # Define loss function and optimizer
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    # Prepare the optimizer
    ## Previous Learning rate : 0.00001
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,betas=(0.9,0.98),eps=1e-6,weight_decay=0.001) # the lr is smaller, more safe for fine tuning to new dataset

    unique_run_id = str(uuid.uuid4())[0:6]
    print(f"Unique Id for this run is : {unique_run_id}")
    datetimenow = datetime.now()
    datetime_fmt = f"{datetimenow.year}_{datetimenow.month}_{datetimenow.day}__{datetimenow.hour}{datetimenow.minute}{datetimenow.second}"

    outputDir = f"logs/{datetime_fmt}/{unique_run_id}"

    num_epochs = FLAGS.num_epochs
    total_loss = None
    Losses = []
    val_losses = []
    for epoch in range(num_epochs):

        model.train(mode=True)
        pbar = tqdm(train_dataloader, total=len(train_dataloader))
        Batch_losses = []
        for batch in pbar:
            if batch is None:
                continue
        
            optimizer.zero_grad()
            
            #images,texts = batch['image'], batch['caption']
            images,texts = batch
            
            #print(images.size(), texts.size())

            images= images.to(device)
            texts = texts.to(device)

            # Forward pass
            logits_per_image, logits_per_text = model(images, texts)

            # Compute loss
            ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
            total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
            
            Batch_losses.append(total_loss.item())

            # Backward pass
            total_loss.backward()
            if device == "cpu":
                optimizer.step()
            else : 
                #convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

            pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss.item():.4f}")


        os.makedirs(f"{outputDir}", exist_ok=True)
        #torch.save(model.state_dict(), f'{outputDir}/HAR_CLIP_final_{datetime_fmt}_{unique_run_id}_{FLAGS.num_epochs}.pth')
        
        Batch_losses = np.array(Batch_losses)
        Losses.append(Batch_losses.mean())

        val_loss = validate_model(model=model,val_dataloader=val_dataloader,loss_img=loss_img,loss_txt=loss_txt ,device=device)
        val_losses.append(val_loss)
        
        saveModel = False
        if len(Losses)>1:
            minLoss = min(Losses)
            if Losses[-1]<=minLoss:
                saveModel = True
        else:
            saveModel = True
                
        
        if saveModel:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss,
                }, f"{outputDir}/HAR_CLIP_final_{datetime_fmt}_{unique_run_id}_{epoch}.pt")
              
    
    os.makedirs(f"{outputDir}", exist_ok=True)

    try:
        # Plot the loss
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, FLAGS.num_epochs + 1), Losses, marker='o', linestyle='-', color='b')
        plt.plot(range(1, FLAGS.num_epochs + 1), val_losses, marker='o', linestyle='-', color='g')
        plt.title(f'Training and validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.grid(True)
        plt.savefig(f"{outputDir}/{unique_run_id}_loss.png")
        # plt.show()
        plt.close()

    except Exception as e:
        print(f"Error plotting acc and loss E: {e}")
    
    
    
    
    
    
    
    