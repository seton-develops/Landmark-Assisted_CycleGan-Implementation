'''
@AUTHOR: Sean Tonthat and Zeting Luan
@DATE: 6/8/22
@DESCRIPTION: Regressor model to generate landmark coordinates from either human image or cartoon image
'''

import torch
import torch.nn as nn
'''
INPUT:
    (128x128x3) image
OUTPUT:
    (,10) vector of predicted landmark coordinates
'''

class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()

        self.nc = 3

        self.main = nn.Sequential(

            nn.Conv2d(self.nc, 20, 5, 1, 0, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(20, 48, 5, 1, 0, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(48, 64, 3, 1, 0, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 80, 3, 1, 0, bias = True),
            nn.ReLU(),

            nn.Flatten(),

            nn.Dropout(p = 0.1),

            nn.Linear(9680, 256),
            nn.ReLU(),

            nn.Dropout(p = 0.1),

            nn.Linear(256, 10),


        )

    def forward(self, input):
        return self.main(input)
