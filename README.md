# Landmark-Assisted-StarGAN
Project to convert images from CelebA dataset to bitemoji avatars.


## Download dataset or our pre-trained models


We have already provided the xlsx file for the landmark coordinates inside the ~/Landmark-Assisted-StarGAN/data/train(val) folder. 
You should replace the trainA, trainB, valA, and valB folder with the folder you downloaded from our link. 

NOTE: PLEASE DELETE ANY FILE inside trainA, trainB, valA, valB, and saved_images folder. Since Github does not allow uploading empty folder, we put a txt file there. 
NOTE: we did not use the val data for validation. The data in valA(B) folder is for testing.

## Training the model


If you want to use a different training setting, feel free to change the hyper-parameters in config.py
If you want to recreate our training process, run:

cd ECE228_FINAL_PROJECT_GROUP24/Landmark-Assisted-StarGAN

Then, run:

python train_first.py

The images generated during the training process is saved in saved_images folder. 

After training with 60 epochs, you need to go to config.py and change the NUM_EPOCHS to 40.

Then, run:

python train.py

## Test and visualize the model


Go to Testing_visualization.ipynb to test and visualize the result. 
We provided 20 images generated from the pre-trained StarGAN model inside the stargan_input and stargan_out folder.

If you set up everything correctly, you should be able to run the cell in sequence without any problem.







## Overview of the project
