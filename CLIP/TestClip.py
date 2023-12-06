
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import numpy as np
from utils.Helpers import CLIPDataset, convert_models_to_fp32
from torch.utils.data import DataLoader
import os

from PIL import Image
import matplotlib.pyplot as plt


import clip
# from transformers import CLIPProcessor, CLIPModel

# Check if cuda is available
use_cuda = torch.cuda.is_available()

# Set proper device based on cuda availability 
device = torch.device("cuda" if use_cuda else "cpu")


model, preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training
checkpoint = torch.load("./weights/HAR_CLIP_final_2023_12_3__2259_ee9c03_95.pt")


# Use these 3 lines if you use default model setting(not training setting) of the clip. For example, if you set context_length to 100 since your string is very long during training, then assign 100 to checkpoint['model_state_dict']["context_length"] 
# checkpoint['model_state_dict']["input_resolution"] = model.input_resolution #default is 224
# checkpoint['model_state_dict']["context_length"] = model.context_length # default is 77
# checkpoint['model_state_dict']["vocab_size"] = model.vocab_size 

model.load_state_dict(checkpoint['model_state_dict'])

DatasetPath = "../train" #FLAGS.dataset_path 
AnnotationFile = "../Training_set.csv" # FLAGS.annotation_file_path
HAR_data = CLIPDataset(ann_file=AnnotationFile,img_preprocessing_fn=preprocess,rootPath=DatasetPath)

text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in HAR_data.classes]).to(device)

import glob
testImages = glob.glob("../test/*.jpg")

model.eval()

cnt = 0
for imgPath in testImages:
    with torch.no_grad():

        PilImage = Image.open(f"{imgPath}")

        image_features = preprocess(PilImage).unsqueeze(0).to(device)
        
        image_attention = model.visual.forward_attention(image_features.type(model.dtype))
        print(image_attention.shape)

        break

        # print(image_features.size(), text_features.size())

        image_features = model.encode_image(image_features)
        text_features = model.encode_text(text_inputs)


        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(5)

        TopKClasses = []
        Scores = []
        for value, index in zip(values, indices):
            print(f"{HAR_data.classes[index]:>16s}: {100 * value.item():.2f}%")
            TopKClasses.append(HAR_data.classes[index])
            Scores.append(round(100 * value.item(),2))

        print()
        
        # maxindex = logits.argmax(dim=-1)
        # predClass = HAR_data.classes[maxindex]

        classDir = f"output/{TopKClasses[0]}"
        imageSavePath = f"{classDir}/{cnt}_{TopKClasses[0]}_{Scores[0]}_{TopKClasses[1]}_{Scores[1]}_{TopKClasses[2]}_{Scores[2]}.jpg"
        os.makedirs(classDir, exist_ok=True)

        PilImage.save(imageSavePath)
        # print(f"Predicted: {predClass} conf: {logits[0][maxindex]}")

        cnt+=1

        # if cnt>10:
        #     break