'''
@AUTHOR: Sean Tonthat
@DATE: 6/8/22
@DESCRIPTION: Helper functions to extract patches that will be input into local discriminators
'''


import cv2
import numpy as np
import torch


def pad_eyes_crop(img_left, img_right):
    '''
    Given two eye patches (left eye, right eye) create a 64x64 image after padding
    
    INPUT: 
        img_left : (np.array) image of left eye patch. Shape is (32x32x3)
        img_right : (np.array) image of right eye patch. Shape is (32x32x3)
    
    OUTPUT: (np.array) 64x64 padded image of the eyes
    '''
    
    #cast to float32 otherwise cv2 won't work
    if not isinstance(img_left[0][0][0], np.float32):
        img_left = img_left.astype('float32')
        img_right = img_right.astype('float32')

    
    #concat the two images together side by side
    img_concat = cv2.hconcat([img_left, img_right])
    
    #pad the image 
    return cv2.copyMakeBorder(img_concat,16,16,0,0,cv2.BORDER_CONSTANT,value=[255,255,255])





def pad_nose_crop(img):
    '''
    Given a nose patch create a 64x64 image after padding
    
    INPUT: 
        img: (np.array) image of mouth patch of shape (28x24x3)
    
    OUTPUT: (np.array) 64x64 padded image of the nose
    '''
    
    #cast to float32 otherwise cv2 won't work
    if not isinstance(img[0][0][0], np.float32):
        img = img.astype('float32')
    
    #pad image
    return cv2.copyMakeBorder(img,18,18,20,20,cv2.BORDER_CONSTANT,value=[255,255,255])





def pad_mouth_crop(img):
    '''
    Given a mouth patch create a 64x64 image after padding
    
    INPUT: 
        img: (np.array) image of mouth patch of shape (23x40x3)
    
    OUTPUT: (np.array) 64x64 padded image of the mouth
    '''
    
    #cast to float32 otherwise cv2 won't work
    if not isinstance(img[0][0][0], np.float32):
        img = img.astype('float32')

    #pad image    
    return cv2.copyMakeBorder(img,21,20,12,12,cv2.BORDER_CONSTANT,value=[255,255,255])





def crop_eyes(image, x_left, y_left, x_right, y_right):
    '''
    Given an image and the coordinates for the eyes (left and right) extract a patch for each eye
    
    INPUT: 
        image - (np.array) Image of shape (128x128x3)
        x_left : (INT) x-coordinate for left eye
        y_left : (INT) y-coordinate for left eye
        x_right : (INT) x-coordinate for right eye
        y_right : (INT) y-coordinate for right eye
    
    OUTPUT: 
        crop_img_left: (np.array) patch for left eye (32x32x3)
        crop_img_right: (np.array) patch for right eye (32x32x3)
    '''
    
    #crop 32 by 32 each eye
    crop_img_left = image[y_left-16:y_left+16, x_left-16:x_left+16]
    crop_img_right = image[y_right-16:y_right+16, x_right-16:x_right+16]
    
    return crop_img_left, crop_img_right





def crop_nose(image, x, y):
    '''
    Given an image and the coordinates for the nose extract the nose patch from the image
    
    INPUT: 
        image: (np.array) single image of shape (128x128x3)
        x : (INT) x-coordinate for center of nose
        y : (INT) y-coordinate for center of nose
    
    OUTPUT: 
        crop_nose: (np.array) patch for left eye (28x24x3)
    '''
    
    #crop 28x24 for nose
    crop_nose = image[y-14:y+14, x-12:x+12]
    return crop_nose






def crop_mouth(image, x_left, y_left, x_right, y_right):
    '''
    Given an image and the coordinates for the mouth extract the mouth patch from the image
    
    INPUT: 
        image: (np.array) single image of shape (128x128x3)
    
    OUTPUT: 
        crop_mouth: (np.array) patch for left eye (23x40x3)
    '''
    
    #23x40 for mouth
    mid_x = round((x_left + x_right) / 2)
    mid_y = round((y_left + y_right) / 2)
    
    crop_mouth = image[ mid_y-12:mid_y+11, mid_x-20:mid_x+20]
    
    return crop_mouth





def create_64x64_patches(image, coord):
    '''
    Given an image and a set of coordinates for the eyes, nose, and mouth return 
    the padded image version of the cropped patches for each of those landmarks.
    This method calls the helper functions above and is called by train.py
    
    INPUT: 
        image: (np.array) single image of shape (128x128x3)
        coord: (1x10) list containing the coordinates
            eyes have indexes 0-3, nose 4-5, mouth 6-9
    OUTPUT: 
        pad_eyes: (torch.tensor) A 64x64 image containing both eye patches
        pad_nose: (torch.tensor) A 64x64 image containing the nose patch
        pad_mouth: (torch.tensor) A 64x64 image containing the mouth patch
    '''
    
    coord = np.array(coord[0])
    
    #Sometimes regressor may turn negative coordinates if image is poor. 
    #In this case, we substitute the coord list with the following
    if np.any(coord < 0):
        coord = np.array([49, 64, 77, 64, 63, 76, 52, 88, 76, 88])
    else:
        coord = coord.astype(int)
        
    
    #switch image shape from 3x128x128 to 128x128x3
    image = image.cpu().detach().numpy()[0]
    image = np.swapaxes(image, 2, 0)
    
    #get cropped images of local landmarks
    cropped_eye_left, cropped_eye_right = crop_eyes(image, coord[0], coord[1], coord[2], coord[3])
    cropped_nose = crop_nose(image,  coord[4], coord[5])
    cropped_mouth = crop_mouth(image, coord[6], coord[7], coord[8], coord[9])
    
    #pad these landmark patches to 64x64 images
    pad_eyes = pad_eyes_crop(cropped_eye_left, cropped_eye_right)
    pad_nose = pad_nose_crop(cropped_nose)
    pad_mouth = pad_mouth_crop(cropped_mouth)
    
    #switch back shape to 3x128x128
    pad_eyes = np.swapaxes(pad_eyes, 2, 0)
    pad_nose = np.swapaxes(pad_nose, 2, 0)
    pad_mouth = np.swapaxes(pad_mouth, 2, 0)
    
    #convert back to tensor
    pad_eyes = torch.from_numpy(pad_eyes)
    pad_nose =  torch.from_numpy(pad_nose)
    pad_mouth =  torch.from_numpy(pad_mouth)
    
    pad_eyes = torch.unsqueeze(pad_eyes, 0)
    pad_nose = torch.unsqueeze(pad_nose, 0)
    pad_mouth = torch.unsqueeze(pad_mouth, 0)
    
    
    return pad_eyes, pad_nose, pad_mouth

