#!/usr/bin/env python
# coding: utf-8

# In[233]:


import pathlib
import os
import openpyxl
import csv
import os
from glob import glob
import pandas as pd
path_to_toon_csv_folder = "/home/zluan/ECE_228_project/cartoonset100k" #place path here
path_to_main_csv = "/home/zluan/ECE_228_project/datasets" #place path of csv file you are writing to here


def read_write_files():
    f = os.path.join(path_to_toon_csv_folder, "*/*.csv") #images are placed in 10 folders, so loop through all
    files = glob(f)
    for file in files:
        flag = 1
        filename = os.path.basename(file).split(".")[0] #image ID, same as the name of the image
        data_attribute_row = []
        data_attribute_row.append(filename)
        with open(file, 'r') as csv_file:
            data = pd.read_csv(csv_file, usecols = [1], header=None) #Only take col with feature values

        for i in range(len(data[1])):
            data_attribute_row.append(str(data[flag][i]))
        data_attribute_mat.append(data_attribute_row) 
        flag += 1
        excel_header = ['ID', 'eye_angle', 'eye_lashes','eye_lid',
                         'chin_length','eyebrow_weight','eyebrow_shape',
                          'eyebrow_thickness','face_shape','facial_hair',
                          'hair','eye_color','face_color','hair_color',
                         'glasses','glasses_color','eye_slant','eyebrow_width', 
                         'eye_eyebrow_distance']
    
    data_attribute_mat.pop(0)
    df = pd.DataFrame(data_attribute_mat, columns = excel_header)
        
    df.to_csv(path_to_main_csv + '/combined_toon_attributes.csv', index=False)



def main():
    read_write_files()
    
    
        
            
if __name__ == '__main__':
    main()            
            
            
            

