'''
@Authors: Zeting Luan & Sean Tonthat

@Purpose: Train CycleGAN model with Local Consistency Loss and Local Landmark Loss

@Note: We used the structure from the official CycleGAN implementation: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
However, no code was copied and only an understanding of the architecture was abstracted.

@Date: 6/8/22
'''

import torch
from dataset import Human2CartoonDataset
import sys
from utils import save_checkpoint, load_checkpoint, calculate_error_norm
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
import numpy as np


#trained models from 1st step of training
'''
We split the training into two phases as the regressor will not be able to predict the landmark locations
while the model is just learning to generate cartoon images.
'''

from model_D import Discriminator
from model_G import Generator
from model_R import Regressor


#added for local discriminator
from patch_extractor import create_64x64_patches
from local_landmark_discriminator import Local_Discriminator
from torchsummary import summary




def train_fn(disc_H, disc_C,
             gen_H, gen_C,
             reg_H, reg_C,
             loader,
             opt_disc, opt_gen, opt_disc_local,
             l1, mse, BCELoss,
             d_scaler, g_scaler,
             nose_human_discrim,
             nose_cartoon_discrim,
             mouth_human_discrim,
             mouth_cartoon_discrim,
             eye_human_discrim,
             eye_cartoon_discrim):


     """
    Trains the CycleGAN w/ Landmark Consistency Loss and Local Discriminator Loss.


    INPUTS/ARGUMENTS:
    - disc_H: global discriminator for human - INPUT: (128,128,3) human image
    - disc_C: global discriminator for catoon - INPUT: (128,128,3) cartoon image

    - gen_H: global generator for human - INPUT: (128,128,3) human image
    - gen_C: global generator for cartoon - INPUT: (128,128,3) cartoon image

    - reg_H: landmark regressor for human. - INPUT: (128,128,3) human image
    - reg_C: landmark regressor for cartoon - INPUT: (128,128,3) cartoon image

    - loader: dataloader for data set

    - opt_disc: optimizer for global discriminator
    - opt_gen: optimizer for global generator
    - opt_disc_local: optimizer for local discriminators

    -l1: L1 Norm loss
    -mse: Mean Squared Error Loss
    -BCELoss - Binary Cross Entropy Loss

    -d_scaler: Instance of torch.cuda.amp.GradScaler for gradient scaling for discriminator
    -g_scaler: Instance of torch.cuda.amp.GradScaler for gradient scaling for generator

    - nose_human_discrim: Local discriminator for human nose patch - INPUT: (64,64, 3) human nose image
    - nose_cartoon_discrim: Local discriminator for cartoon nose patch - INPUT: (64,64, 3) cartoon nose image
    - mouth_human_discrim: Local discriminator for human mouth patch - INPUT: (64,64, 3) human mouth image
    - mouth_cartoon_discrim: Local discriminator for cartoon mouth patch - INPUT: (64,64, 3) cartoon mouth image
    - eye_human_discrim: Local discriminator for human eyes patch - INPUT: (64,64, 3) human eyes image
    - eye_cartoon_discrim: Local discriminator for cartoon eyes patch - INPUT: (64,64, 3) cartoon eyes image



    RETURNS:
    None
    """



    H_reals = 0
    H_fakes = 0
    C_reals = 0
    C_fakes = 0
    landmark_human_success_flag = 0
    landmark_cartoon_success_flag = 0
    loop = tqdm(loader, leave=True)


    for idx, (human, cartoon, landmark_human, landmark_cartoon) in enumerate(loop):
        cartoon = cartoon.to(config.DEVICE)
        human = human.to(config.DEVICE)
        landmark_human = landmark_human.to(config.DEVICE)
        landmark_cartoon = landmark_cartoon.to(config.DEVICE)


        with torch.cuda.amp.autocast():

            """TRAINING OF GLOBAL DISCRIMINATORS"""
            # Train Discriminators for human
            fake_human = gen_H(cartoon.detach())
            D_H_real = disc_H(human.detach())
            D_H_fake = disc_H(fake_human.detach())
            H_reals += D_H_real.mean().item()
            H_fakes += D_H_fake.mean().item()
            # losses
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss


            # Train Discriminators for cartoon
            fake_cartoon = gen_C(human.detach())
            D_C_real = disc_C(cartoon.detach())
            D_C_fake = disc_C(fake_cartoon.detach())
            C_reals += D_C_real.mean().item()
            C_fakes += D_C_fake.mean().item()
            #losses
            D_C_real_loss = mse(D_C_real, torch.ones_like(D_C_real))
            D_C_fake_loss = mse(D_C_fake, torch.zeros_like(D_C_fake))
            D_C_loss = D_C_real_loss + D_C_fake_loss
             """END OF TRAINING OF GLOBAL DISCRIMINATORS"""




            """CALCULATE LANDMARK CONSISTENCY LOSS"""

            #Get landmark coordinate predictions from regressor
            landmark_human_pred = reg_H((fake_human.detach()*0.5+0.5)*255)
            landmark_cartoon_pred = reg_C((fake_cartoon.detach()*0.5+0.5)*255)

            #Calculate the landmark consistency loss between real and generated human
            landmark_human_error_norm = calculate_error_norm(landmark_human.detach().cpu().numpy(), landmark_human_pred.detach().cpu().numpy())
            landmark_human_loss_arry = np.mean(landmark_human_error_norm) * 100
            landmark_human_loss = torch.tensor(landmark_human_loss_arry)

            #Calculate the landmark consistency loss between real and generated cartoon
            landmark_cartoon_error_norm = calculate_error_norm(landmark_cartoon.detach().cpu().numpy(), landmark_cartoon_pred.detach().cpu().numpy())
            landmark_cartoon_loss_arry = np.mean(landmark_cartoon_error_norm) * 100
            landmark_cartoon_loss = torch.tensor(landmark_cartoon_loss_arry)

            #Combine the two losses
            landmark_loss = landmark_human_loss + landmark_cartoon_loss
            """END OF LANDMARK CONSISTENCY LOSS"""




            '''EXTRACT PATCHES OF LOCAL LANDMARKS FOR LOCAL LANDMARK DISCRIMINATORS'''
            #get patches for training data
            #create_64x64_patches will get the local patches and pad them into 64x64 images
            h_eyes, h_mouth, h_nose = create_64x64_patches(human, landmark_human.detach().cpu().numpy())
            c_eyes, c_mouth, c_nose = create_64x64_patches(cartoon, landmark_cartoon.detach().cpu().numpy())

            #get patches for generated images
            gen_h_eyes, gen_h_mouth, gen_h_nose = create_64x64_patches(fake_human, landmark_human_pred.detach().cpu().numpy())
            gen_c_eyes, gen_c_mouth, gen_c_nose = create_64x64_patches(fake_cartoon, landmark_cartoon_pred.detach().cpu().numpy())
            '''END OF PATCH EXTRACTION'''




            '''INPUT PATCHES INTO LOCAL DISCRIMINATORS'''
            #Feed patches of nose into local nose discriminators
            pred_human_nose = nose_human_discrim(h_nose)
            pred_cartoon_nose = nose_cartoon_discrim(c_nose)
            pred_gen_human_nose = nose_human_discrim(gen_h_nose)
            pred_gen_cartoon_nose = nose_cartoon_discrim(gen_c_nose)

            #Feed patches of mouth into local mouth discriminators
            pred_human_mouth = mouth_human_discrim(h_mouth)
            pred_cartoon_mouth = mouth_cartoon_discrim(c_mouth)
            pred_gen_human_mouth = mouth_human_discrim(gen_h_mouth)
            pred_gen_cartoon_mouth = mouth_cartoon_discrim(gen_c_mouth)

            #Feed patches of eye into local eye discriminators
            pred_human_eyes = eye_human_discrim(h_eyes)
            pred_cartoon_eyes = eye_cartoon_discrim(c_eyes)
            pred_gen_human_eyes = eye_human_discrim(gen_h_eyes)
            pred_gen_cartoon_eyes = eye_cartoon_discrim(gen_c_eyes)

            '''END OF PATCH INSERTION'''




            '''CALCULATE LOCAL DISCRIMINATOR LOSS'''
            #Get Binary Cross Entropy Loss for the noses from the local nose discriminators
            nose_human_loss =        BCELoss(pred_human_nose , torch.ones_like(pred_human_nose))
            nose_cartoon_loss =      BCELoss(pred_cartoon_nose , torch.ones_like(pred_cartoon_nose))
            gen_nose_human_loss =    BCELoss(pred_gen_human_nose , torch.zeros_like(pred_gen_human_nose))
            gen_nose_cartoon_loss =  BCELoss(pred_gen_cartoon_nose , torch.zeros_like(pred_gen_cartoon_nose))

            #Get Binary Cross Entropy Loss for the mouths from the local mouths discriminators
            mouth_human_loss =       BCELoss(pred_human_mouth , torch.ones_like(pred_human_mouth))
            mouth_cartoon_loss =     BCELoss(pred_cartoon_mouth , torch.ones_like(pred_cartoon_mouth))
            gen_mouth_human_loss =   BCELoss(pred_gen_human_mouth , torch.zeros_like(pred_gen_human_mouth))
            gen_mouth_cartoon_loss = BCELoss(pred_gen_cartoon_mouth , torch.zeros_like(pred_gen_cartoon_mouth))

            #Get Binary Cross Entropy Loss for the eyes from the local eyes discriminators
            eyes_human_loss =        BCELoss(pred_human_eyes , torch.ones_like(pred_human_eyes))
            eyes_cartoon_loss =      BCELoss(pred_cartoon_eyes , torch.ones_like(pred_cartoon_eyes))
            gen_eyes_human_loss =    BCELoss(pred_gen_human_eyes , torch.zeros_like(pred_gen_human_eyes))
            gen_eyes_cartoon_loss =  BCELoss(pred_gen_cartoon_eyes , torch.zeros_like(pred_gen_cartoon_eyes))

            #Sum all the losses
            local_landmark_loss = sum([nose_human_loss, nose_cartoon_loss, gen_nose_human_loss, gen_nose_cartoon_loss, mouth_human_loss, mouth_cartoon_loss, gen_mouth_human_loss, gen_mouth_cartoon_loss, eyes_human_loss, eyes_cartoon_loss, gen_eyes_human_loss,  gen_eyes_cartoon_loss])

            #Multiply the loss by a weight. Weight can be found in the config file
            local_landmark_loss = config.LAMBDA_LANDMARK_LOCAL * local_landmark_loss

            '''END OF CALCULATING LOSS'''



            # CALCULATE THE DISCRIMINATOR LOSS FOR GLOBAL and LOCAL DISCRIMINATORS
            #Because we are adding up human and cartoon loss, we average them out by dividing by two
            D_loss = (D_H_loss + D_C_loss + local_landmark_loss) / 2




        '''PERFORM BACKPROPOGATION'''
        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        opt_disc_local.zero_grad()
        local_landmark_loss.backward()
        opt_disc_local.step()
        '''END OF PERFORM BACKPROPOGATION'''


        # Train Generators H and C
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_H_fake = disc_H(fake_human)
            D_C_fake = disc_C(fake_cartoon)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_C = mse(D_C_fake, torch.ones_like(D_C_fake))

            # cycle loss
            cycle_cartoon = gen_C(fake_human)
            cycle_human = gen_H(fake_cartoon)
            cycle_cartoon_loss = l1(cartoon, cycle_cartoon)
            cycle_human_loss = l1(human, cycle_human)


            # add all togethor
            G_loss = (
                loss_G_C
                + loss_G_H
                + cycle_cartoon_loss * config.LAMBDA_CYCLE
                + cycle_human_loss * config.LAMBDA_CYCLE
                + landmark_loss * config.LAMBDA_LANDMARK
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(fake_human*0.5+0.5, f"saved_images/human_{idx}.png")
            save_image(fake_cartoon*0.5+0.5, f"saved_images/cartoon_{idx}.png")

        loop.set_postfix(C_real=C_reals/(idx+1), C_fake=C_fakes/(idx+1),
                         G_loss=G_loss.item()/(idx+1),
                         D_loss=D_loss.item()/(idx+1),
                         S_H = landmark_human_success_flag, # successful match
                         S_C = landmark_cartoon_success_flag
                                 )



def main():

    '''
    Calls the train() method above
    Instantiates the parameters for train()
    '''

    #Instantiate the regressors, discriminators, and generators
    disc_H = Discriminator(in_channels=3).to(config.DEVICE) #global discriminators
    disc_C = Discriminator(in_channels=3).to(config.DEVICE)

    gen_C = Generator(img_channels=3, num_residuals=9).to(config.DEVICE) #generators
    gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)

    reg_C = Regressor().to(config.DEVICE) #landmark regressor cartoon
    reg_C.load_state_dict(torch.load('cartoon_torch.pt'))

    reg_H = Regressor().to(config.DEVICE) #landmark regressor human
    reg_H.load_state_dict(torch.load('human_torch.pt'))



    #Local Discriminators
    nose_human_discrim = Local_Discriminator()
    nose_cartoon_discrim = Local_Discriminator()
    mouth_human_discrim = Local_Discriminator()
    mouth_cartoon_discrim = Local_Discriminator()
    eye_human_discrim = Local_Discriminator()
    eye_cartoon_discrim = Local_Discriminator()



    #optimizer for local landmark discriminators
    opt_disc_local = optim.Adam(
        list(nose_human_discrim.parameters())
        + list(nose_cartoon_discrim.parameters())
        + list(mouth_human_discrim.parameters())
        + list(mouth_cartoon_discrim.parameters())
        + list(eye_human_discrim.parameters())
        + list(eye_cartoon_discrim.parameters()),

        lr=config.LEARNING_RATE,

        betas=(0.5, 0.999),
    )



    #optimizer for global discriminators
    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_C.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_C.parameters()) + list(gen_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )



    #Instantiate various losses
    BCELoss = nn.BCELoss()
    L1 = nn.L1Loss()
    mse = nn.MSELoss()



    #If config.LOAD_MODEL is TRUE, then load previous checkpoints for all trainable models
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_G_H, gen_H, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_G_C, gen_C, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_D_H, disc_H, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_D_C, disc_C, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_D_NOSE_H_LOCAL, nose_human_discrim, opt_disc_local, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_D_NOSE_C_LOCAL, nose_cartoon_discrim, opt_disc_local, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_D_MOUTH_H_LOCAL, mouth_human_discrim, opt_disc_local, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_D_MOUTH_C_LOCAL, mouth_cartoon_discrim, opt_disc_local, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_D_EYE_H_LOCAL, eye_human_discrim, opt_disc_local, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_D_EYE_C_LOCAL, eye_cartoon_discrim, opt_disc_local, config.LEARNING_RATE,
        )



    #NOTE: no validation set is required since we are dealing with GANs

    #instance of our own dataloader imported from dataset.py
    dataset = Human2CartoonDataset(
        root_human=config.TRAIN_DIR+"/trainA",
        root_cartoon=config.TRAIN_DIR+"/trainB",
        root_landmarks_human=config.TRAIN_DIR+"/trainA_human_landmarks.xlsx",
        root_landmarks_cartoon=config.TRAIN_DIR+"/trainB_cartoon_landmarks.xlsx",
        transform=config.transforms
    )


    #Instance of pytorch dataloader
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()


    #Begin training
    for epoch in range(config.NUM_EPOCHS):
        print("EPOCH : ", epoch)
        train_fn(disc_H, disc_C,
                 gen_H, gen_C,
                 reg_H, reg_C,
                 loader,
                 opt_disc, opt_gen, opt_disc_local,
                 L1, mse, BCELoss,
                 d_scaler, g_scaler,
                 nose_human_discrim,
                 nose_cartoon_discrim,
                 mouth_human_discrim,
                 mouth_cartoon_discrim,
                 eye_human_discrim,
                 eye_cartoon_discrim)

        #if config.SAVE_MODEL is TRUE, then save model checkpoint
        if config.SAVE_MODEL:
            save_checkpoint(gen_H, opt_gen, filename=config.CHECKPOINT_G_H)
            save_checkpoint(gen_C, opt_gen, filename=config.CHECKPOINT_G_C)
            save_checkpoint(disc_H, opt_disc, filename=config.CHECKPOINT_D_H)
            save_checkpoint(disc_C, opt_disc, filename=config.CHECKPOINT_D_C)
            save_checkpoint(nose_human_discrim, opt_disc_local, filename=config.CHECKPOINT_D_NOSE_H_LOCAL)
            save_checkpoint(nose_cartoon_discrim, opt_disc_local, filename=config.CHECKPOINT_D_NOSE_C_LOCAL)
            save_checkpoint(mouth_human_discrim, opt_disc_local, filename=config.CHECKPOINT_D_MOUTH_H_LOCAL)
            save_checkpoint(mouth_cartoon_discrim, opt_disc_local, filename=config.CHECKPOINT_D_MOUTH_C_LOCAL)
            save_checkpoint(eye_human_discrim, opt_disc_local, filename=config.CHECKPOINT_D_EYE_H_LOCAL)
            save_checkpoint(eye_cartoon_discrim, opt_disc_local, filename=config.CHECKPOINT_D_EYE_C_LOCAL)






if __name__ == "__main__":
    main()
