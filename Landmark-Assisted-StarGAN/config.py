'''
@AUTHOR:  Zeting Luan
@DATE: 6/8/22
@DESCRIPTION: the config file that contains hyper-parameters, directory, and other options
Users can change the True/False setting in LOAD_MODEL or SAVE_MODEL based on the need.
Users can also change the hyper-parameters for different purposes.
'''

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "/home/zluan/ECE228_FINAL_PROJECT_GROUP24-main/Landmark-Assisted-StarGAN/data/train"
VAL_DIR = "/home/zluan/ECE228_FINAL_PROJECT_GROUP24-main/Landmark-Assisted-StarGAN/data/val"
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_CYCLE = 10
LAMBDA_LANDMARK = 1
LAMBDA_LANDMARK_LOCAL = 0.3
NUM_WORKERS = 4
NUM_EPOCHS = 50
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_G_H = "G_H.pth.tar"
CHECKPOINT_G_C = "G_C.pth.tar"
CHECKPOINT_D_H = "D_H.pth.tar"
CHECKPOINT_D_C = "D_C.pth.tar"
CHECKPOINT_D_NOSE_H_LOCAL = "D_NOSE_H_LOCAL.pth"
CHECKPOINT_D_NOSE_C_LOCAL = "D_NOSE_C_LOCAL.pth"
CHECKPOINT_D_MOUTH_H_LOCAL = "D_MOUTH_H_LOCAL.pth"
CHECKPOINT_D_MOUTH_C_LOCAL = "D_MOUTH_C_LOCAL.pth"
CHECKPOINT_D_EYE_H_LOCAL = "D_EYE_H_LOCAL.pth"
CHECKPOINT_D_EYE_C_LOCAL = "D_EYE_C_LOCAL.pth"

transforms = A.Compose(
    [
        A.Resize(width=128, height=128),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)
