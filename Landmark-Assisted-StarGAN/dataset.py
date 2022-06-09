'''
@Authors: Zeting Luan & Sean Tonthat

@Purpose: Load image and landmark data from the directory for both human and cartoon

@Discription: The dataloader that read the human and cartoon images from directory as well as formatting the
landmark coordinate so that it can match with the specific image.

@Date: 6/8/22
'''

from PIL import Image
import os
from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
'''
INPUT:
    root_human: the directory to human dataset. If we train the model, it should be in ~/data/train/trainA
    root_cartoon: the directory to cartoon dataset. If we train the model, it should be in ~/data/train/trainB
    root_landmarks_human: the directory to human landmark xlsx file. If we train the model,i t should be in ~/data/train
    root_landmarks_cartoon: the directory to human cartoon xlsx file. If we train the model,i t should be in ~/data/train
    transform: If we transform the image
OUTPUT:
    human_img: (128,128,3) image from CelebA dataset
    cartoon_img: (128,128,3) image from bitmoji dataset
    landmarks_human: (,10) vector of landmark coordinate from the human xlsx file
    landmarks_cartoon: (,10) vector of landmark coordinate from the cartoon xlsx file
'''
class Human2CartoonDataset(Dataset):
    def __init__(self, root_human, root_cartoon, root_landmarks_human, root_landmarks_cartoon, transform=None):
        '''INSTANTIATES THE ROOT FOR HUMAN AND CARTOON'''
        self.root_cartoon = root_cartoon
        self.root_human = root_human
        self.transform = transform
        '''END OF INSTANTIATES THE ROOT FOR HUMAN AND CARTOON'''

        '''READ LANDMARK FOR BOTH HUMAN AND CARTOON'''
        self.land_human = pd.read_excel(root_landmarks_human)
        self.land_human = pd.concat([self.land_human.columns.to_frame().T,self.land_human], ignore_index=True)
        self.land_human.columns = ['image_ID', 'lefteye_x','lefteye_y','righteye_x', 'righteye_y', 'nose_x', 'nose_y', 'leftmouth_x', 'leftmouth_y', 'rightmouth_x', 'rightmouth_y']

        self.land_cartoon = pd.read_excel(root_landmarks_cartoon)
        self.land_cartoon = pd.concat([self.land_cartoon.columns.to_frame().T,self.land_cartoon], ignore_index=True)
        self.land_cartoon.columns = self.land_human.columns
        '''END OF READ LANDMARK FOR BOTH HUMAN AND CARTOON'''

        '''READ HUMAN AND CARTOON IMAGE'''
        self.cartoon_images = os.listdir(root_cartoon)
        self.human_images = os.listdir(root_human)
        '''END OF READ HUMAN AND CARTOON IMAGE'''

        '''SPECIFY THE LENGTH OF EACH DATASET'''
        self.length_dataset = max(len(self.cartoon_images), len(self.human_images))
        self.cartoon_len = len(self.cartoon_images)
        self.human_len = len(self.human_images)
        '''END OF SPECIFY THE LENGTH OF EACH DATASET'''

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        '''SPECIFY THE PATH AND THE NAME OF THE IMAGE FOR HUMAN AND CARTOON'''
        cartoon_name = self.cartoon_images[index % self.cartoon_len]
        cartoon_path = os.path.join(self.root_cartoon, cartoon_name)

        human_name = self.human_images[index % self.human_len]
        human_path = os.path.join(self.root_human, human_name)
        '''END OF SPECIFY THE PATH AND THE NAME OF THE IMAGE FOR HUMAN AND CARTOON'''

        '''FORMATTING THE LANDMARK COORDINATE SO THAT IT CAN MATCH WITH THE IMAGE LOADED'''
        land_cartoon_arr = self.land_cartoon.loc[self.land_cartoon.index[self.land_cartoon['image_ID']==cartoon_name]].values.tolist()
        landmark_cartoon_list = list(map(float,land_cartoon_arr[0][1:]))
        landmarks_cartoon = torch.tensor(np.array(landmark_cartoon_list))

        land_human_arr = self.land_human.loc[self.land_human.index[self.land_human['image_ID']==human_name]].values.tolist()
        landmark_human_list = list(map(float,land_human_arr[0][1:]))
        landmarks_human = torch.tensor(np.array(landmark_human_list))
        '''END OF FORMATTING THE LANDMARK COORDINATE SO THAT IT CAN MATCH WITH THE IMAGE LOADED'''


        cartoon_img = np.array(Image.open(cartoon_path).convert("RGB"))
        human_img = np.array(Image.open(human_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=cartoon_img, image0=human_img)
            cartoon_img = augmentations["image"]
            human_img = augmentations["image0"]

        return human_img, cartoon_img, landmarks_human, landmarks_cartoon
