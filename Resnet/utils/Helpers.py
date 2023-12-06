import torch.utils.data as data
from PIL import Image
import os
import torch
import torchvision.transforms as transforms 
import numpy as np
from torchvision.io import read_image
import clip

# Function to convert model's parameters to FP32 format
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

class HARDataloader(data.Dataset):
    """ HAR dataloader """

    def __init__(self, ann_file, rootPath="./", seed=123):

        labelWiseImages = {}
        self.labels = []
        self.images = []

        self.transform = transforms.Compose([ 
            # transforms.PILToTensor(),
            transforms.Resize((224,224), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]) 

        with open(f"{ann_file}") as f:
            lines = f.readlines()

            for idx,line in enumerate(lines):
                if idx>0:
                    line = line.strip()
                    imageFilename, imagelabel = line.split(",")
                    self.labels.append(imagelabel)

                    ImageFullPath =  os.path.join(rootPath,imageFilename)
                    self.images.append(ImageFullPath)

                    if not imagelabel in labelWiseImages:
                        labelWiseImages[imagelabel] = []

                    # labelWiseImages[imagelabel].append(imageFilename)

        self.classes = list(labelWiseImages.keys())
        self.classes_zeros_list = [0 for k in self.classes]

    def create_one_hot_encoding(self, label):
        index_of_label = self.classes.index(label)
        zerosList = self.classes_zeros_list.copy()
        zerosList[index_of_label] = 1
        return zerosList

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target)
        """
        oneHotCodedList = self.create_one_hot_encoding(self.labels[index])

        # tensorImg = read_image(f"{self.images[index]}")
        Img = Image.open(f"{self.images[index]}")
        # img_matrix = np.array(Img).astype(np.float64)
        return self.transform(Img), torch.Tensor(oneHotCodedList)

    def __len__(self):
        return len(self.images)
    

class CLIPDataset(torch.utils.data.Dataset):
    """ HAR dataloader """

    def __init__(self, ann_file, img_preprocessing_fn, rootPath="./", seed=123, ):

        labelWiseImages = {}
        self.labels = []
        self.images = []

        self.img_preprocessing_fn = img_preprocessing_fn

        with open(f"{ann_file}") as f:
            lines = f.readlines()

            for idx,line in enumerate(lines):
                if idx>0:
                    line = line.strip()
                    imageFilename, imagelabel = line.split(",")
                    self.labels.append(imagelabel)

                    ImageFullPath =  os.path.join(rootPath,imageFilename)
                    self.images.append(ImageFullPath)

                    if not imagelabel in labelWiseImages:
                        labelWiseImages[imagelabel] = []

                    # labelWiseImages[imagelabel].append(imageFilename)

        self.classes = list(labelWiseImages.keys())
        # self.classes_zeros_list = [0 for k in self.classes]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target)
        """
        image, caption = None, None

        if os.path.exists(self.images[index]):
            Img = Image.open(f"{self.images[index]}")
            image = self.img_preprocessing_fn(Img)
            caption = clip.tokenize(["A photo of a person " +  self.labels[index]]) 
            return image, caption
                
        return None, None

    def __len__(self):
        return len(self.images)