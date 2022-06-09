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

def train_fn(disc_H, disc_C,
             gen_H, gen_C,
             reg_H, reg_C,
             cdisc_H, cdisc_C,
             loader,
             opt_disc, opt_gen, opt_reg,
             l1, mse,
             d_scaler, g_scaler):
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

-l1: L1 Norm loss
-mse: Mean Squared Error Loss
-BCELoss - Binary Cross Entropy Loss

-d_scaler: Instance of torch.cuda.amp.GradScaler for gradient scaling for discriminator
-g_scaler: Instance of torch.cuda.amp.GradScaler for gradient scaling for generator




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
        # Train Discriminators H and C

        with torch.cuda.amp.autocast():
            """TRAINING OF GLOBAL DISCRIMINATORS"""
            # Train Discriminators for human
            fake_human = gen_H(cartoon)
            D_H_real = disc_H(human)
            D_H_fake = disc_H(fake_human.detach())
            H_reals += D_H_real.mean().item()
            H_fakes += D_H_fake.mean().item()
            # losses
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss


            # Train Discriminators for cartoon
            fake_cartoon = gen_C(human)
            D_C_real = disc_C(cartoon)
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

            # CALCULATE THE DISCRIMINATOR LOSS FOR GLOBAL DISCRIMINATOR
            #Because we are adding up human and cartoon loss, we average them out by dividing by two
            D_loss = (D_H_loss + D_C_loss)/2 #+ C_D_loss



        '''PERFORM BACKPROPOGATION'''
        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()
        '''END OF PERFORM BACKPROPOGATION'''

        # Train Generators H and C
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_H_fake = disc_H(fake_human)
            #C_D_H_fake = cdisc_H(fail_img_human.detach())
            D_C_fake = disc_C(fake_cartoon)
            #C_D_C_fake = cdisc_C(fail_img_cartoon.detach())
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))#+mse(C_D_H_fake, torch.zeros_like(C_D_H_fake))
            loss_G_C = mse(D_C_fake, torch.ones_like(D_C_fake))#+mse(C_D_C_fake, torch.zeros_like(C_D_C_fake))

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
            save_image(cartoon*0.5+0.5, f"saved_images/human_{idx}_input.png")
            save_image(human*0.5+0.5, f"saved_images/cartoon_{idx}_input.png")
            save_image(fake_human*0.5+0.5, f"saved_images/human_{idx}.png")
            save_image(fake_cartoon*0.5+0.5, f"saved_images/cartoon_{idx}.png")

        loop.set_postfix(H_real=H_reals/(idx+1), C_real=C_reals/(idx+1),
                         G_loss=G_loss.item()/(idx+1),
                         D_loss=D_loss.item()/(idx+1),
                         S_H = landmark_human_success_flag, # successful match
                         S_C = landmark_cartoon_success_flag,
                         L = landmark_loss.item() * config.LAMBDA_LANDMARK/(idx+1),
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




    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_C.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    opt_reg = optim.Adam(
        list(reg_H.parameters()) + list(reg_C.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    opt_gen = optim.Adam(
        list(gen_C.parameters()) + list(gen_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    #Instantiate various losses
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
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        print('Epoch : ', epoch)
        train_fn(disc_H, disc_C,
                 gen_H, gen_C,
                 reg_H, reg_C,
                 cdisc_H, cdisc_C,
                 loader,
                 opt_disc, opt_gen, opt_reg,
                 L1, mse,
                 d_scaler, g_scaler)


        #if config.SAVE_MODEL is TRUE, then save model checkpoint
        if config.SAVE_MODEL:
            save_checkpoint(gen_H, opt_gen, filename=config.CHECKPOINT_G_H)
            save_checkpoint(gen_C, opt_gen, filename=config.CHECKPOINT_G_C)
            save_checkpoint(disc_H, opt_disc, filename=config.CHECKPOINT_D_H)
            save_checkpoint(disc_C, opt_disc, filename=config.CHECKPOINT_D_C)




if __name__ == "__main__":
    main()
