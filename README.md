# Landmark-Assisted-StarGAN
Project to convert images from CelebA dataset to bitemoji avatars.


## Overview
NOTE: PLEASE DELETE ANY FILE INSIDE `~/trainA`, `~/trainB`, `~/valA`, `~/valB`, and `~/saved_images` folder. Since Github does not allow uploading empty folder, we put an empty txt file there. 

Inside the `Landmark-Assisted-StarGAN` folder:

`~/data` contains the training, testing data, and the landmark coordinate xlsx file we provided.

`~/saved_images` contains the images generated during the training process.

`~/stargan_input` contains the selected input images from CelebA dataset for testing purpose.

`~/stargan_output` contains the selected output images generated by the pre-trained StarGAN model for testing purpose.

`Testing_visualization.ipynb` is for testing the model you trained or the pre-trained model we provided.

`cartoon_torch.pt` is the pre-trained regressor model for cartoon.

`config.py` contains the hyper-parameters, number of epochs, options for load/save models, and etc. You can always change the values to your demand. 

`dataset.py` is the dataloader we created specific for this task.

`human_torch.pt` is the pre-trained regressor model for human.

`local_landmark_discriminator.py` is the local discriminator model.

`model_D.py` is the discriminator model from https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/GANs/CycleGAN.

`model_G.py` is the generator model from https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/GANs/CycleGAN.

`model_R.py` is the regressor model.

`patch_extractor.py` contains functions that crop the images into patches.

`train.py` is the second training stage set up for our model (CycleGAN + landmark consistency loss + local discriminator).

`train_first.py` is the first training set up for our model (CycleGAN + landmark consistency loss).

`utils.py` contains the utils functions to save/load models and calculate the landmark consistency loss.

## Download dataset or our pre-trained models

https://drive.google.com/drive/folders/1b2HdQjdiX-RCQDwpKEbx4qdLChmZARzO?usp=sharing

### dataset
Download `data.zip` in the Google Drive link we provided above.
Unzip it and replace the `~/Landmark-Assisted-StarGAN/data` folder. 

The `~/data` folder should have the structure:
  data---train---trainA--- i.jpg
     |       |  
     |        ---trainB---j.jpg
     |       |
     |        ---trainA_human_landmarks.xlsx
     |       |
     |        ---trainB_cartoon_landmarks.xlsx
     | 
      ---  val ---valA--- k.jpg
             |
              ---valB--- m.jpg
             |
              ---validA_human_landmarks.xlsx
             |
              ---validB_cartoon_landmarks.xlsx
              
              
              
.
├── data
│    ├── train                    # Test files (alternatively `spec` or `tests`)
│    │   ├── benchmarks          # Load and stress tests
│        ├── integration         # End-to-end, integration tests (alternatively `e2e`)
│        └── unit                # Unit tests
└── ...            

```bash
data
   ├── train
   │       ├── trainA
   │       │        └─── *.jpg
   │       ├── trainB
   │       │        └─── *.jpg
   │       ├── trainA_human_landmarks.xlsx
   │       └─── trainB_cartoon_landmarks.xlsx
   │
   └─── val
          ├── valA
          │      └─── *.jpg
          ├── valB
          │      └─── *.jpg
          ├── valA_human_landmarks.xlsx
          └─── valB_cartoon_landmarks.xlsx

```


If you just replace the `~/data` folder, there should not be any problem with the folder structure.            

NOTE: we did not use the val data for validation. The data in `~/valA(B)` folder is for testing.

### pre-trained models
If you want to use the pre-trained models.
Download `pre-trained-models.zip` in the same Google Drive link.
Please put evey pre-trained model inside the `~/Landmark-Assisted-StarGAN` folder. You should have total of 10 `.pth.tar` files.

In fact, you only need the generator model `G_H.pth.tar` to test and visualize, but we provided all the models anyway so you can also continue training or use them to your demand. 



## Training the model
IMPORTANT: GO TO `config.py` FIRST. Change `TRAIN_DIR` and `VAL_DIR` to absolute path, for example : `/home/zluan/ECE228_FINAL_PROJECT_GROUP24-main/Landmark-Assisted-StarGAN/data/train`. (relative path might lead to "[Errno 2] No such file or directory:")

If you want to use a different training setting, feel free to change the hyper-parameters in `config.py`.

If you want to recreate our training process, run:

`cd ECE228_FINAL_PROJECT_GROUP24/Landmark-Assisted-StarGAN`

Then, run:

`python train_first.py`

The images generated during the training process is saved in saved_images folder. 

After training with 60 epochs, feel free to test and visualize this model at this stage. Please refer to `Test and visualize the model` section. 


If you want to continue, you need to go to `config.py` and change the `NUM_EPOCHS` to 40 and change `LOAD_MODEL` to True.

Then, run:

`python train.py`

## Test and visualize the model


Go to Testing_visualization.ipynb to test and visualize the result. 
We provided 20 images selected from CelebA dataset and 20 images generated from the pre-trained StarGAN model inside the stargan_input and stargan_out folder.

If you set up everything correctly, you should be able to run the cell in sequence without any problem.










