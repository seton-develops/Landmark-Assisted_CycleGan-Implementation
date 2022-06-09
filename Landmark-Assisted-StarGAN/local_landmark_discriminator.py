'''
@AUTHOR: Sean Tonthat
@DATE: 6/8/22
@DESCRIPTION: This file contains the local landmark discriminator. Each landmark patch will use an instance of this model
The model takes in 64x64x3 images. 

This file will be called in train.py
'''


import torch
import torch.nn as nn




class Local_Discriminator(nn.Module):
    def __init__(self):
        super(Local_Discriminator, self).__init__()
        
        self.num_channels = 3
        self.out_chan = 32
        
        self.main = nn.Sequential(
            nn.Conv2d(self.num_channels, self.out_chan, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(self.out_chan, self.out_chan * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.out_chan * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.out_chan * 2, self.out_chan * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.out_chan * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.out_chan * 4, self.out_chan * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.out_chan * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.out_chan * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
            nn.Flatten()
        )

    def forward(self, input):
        return self.main(input)

