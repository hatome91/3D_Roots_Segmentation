#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:34:50 2020

@author: fatome
"""
import roots_utils as ru
from datetime import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


######################
# Global variable
######################

pathToFile = '/home/fatome/GitHub_Repos/3D_Roots_Segmentation/Tomo/Beans_Small_Real_Soil.raw'
pathToMask = '/home/fatome/GitHub_Repos/3D_Roots_Segmentation/mask.png'
dims = (1610,1610,1376)
subdims = (115,115,86)
nblock = np.divide(dims, subdims).astype(np.int)
saveto = 'Tomo/Real_Soil_thresh_Slices.raw'

#####################
# Slices to be shown
#####################
np.random.seed(91)

sl = []
sl.append(np.random.randint(0, int(dims[2] / 3)))
sl.append(np.random.randint(int(dims[2] / 3), int(2 * dims[2] / 3)))
sl.append(np.random.randint(int(dims[2] * 2 / 3), dims[2]))


###################################
# Check if the outputfile exists
###################################
if os.path.exists(saveto):
    os.remove(saveto)


####################
# Load the mask
####################
mask = cv2.imread(pathToMask, 0)
mask = (np.array(mask)/255).astype(np.uint8)

######################
# Main 
######################
if __name__ == '__main__':
    
    # imgGen = ru.load_slice(pathToFile, dims)
    
    # for i, img in enumerate(imgGen):
    #     continue
    # print(i+1)
    
    imgGen = ru.load_slice(pathToFile, dims)
    
    for n, img in enumerate(imgGen):
        if n == 464:
            plt.imsave('Slice_'+ str(n) + '.png', img * mask, cmap='gray')
            break
            
        # thresh = ru.preprocessing(img)
        # ru.save_to_raw(saveto, thresh)
        
        
        # if n % sl[0] == 0 and n != 0:
        #     plt.imsave('Tomo/Slice_' + str(n) + '.png', img, cmap='gray')
        #     plt.imsave('Tomo/Thresh_' + str(n) + '.png', thresh, cmap='gray')
        #     sl.pop(0)
        # if n % 100 == 0 and n!= 0:
        #     break
    
    
    
    # volGen = ru.load_region(pathToFile, dims, subdims)
    
    # block = np.zeros((dims[0], dims[1], subdims[2]))
    
    # row = 0
    # col = 0
    
    # for n, vol in enumerate(volGen):
        
    #     print('Processing block', n, 'out of ', nblock[0]*nblock[1]*nblock[2])
    #     thresh = ru.preprocessing(vol)
        
    #     print('Arranging the blocks into a consistent volume')
    #     if n % nblock[1] != 0 or col == 0:
    #         block[row*subdims[0]:(row+1)*subdims[0], col*subdims[1]:(col+1)*subdims[1], :] = thresh
    #         col += 1
    #     else:
    #         row += 1
    #         col = 0
    #         block[row*subdims[0]:(row+1)*subdims[0], col*subdims[1]:(col+1)*subdims[1], :] = thresh
    #         col += 1
        
    #     if (n + 1) % (nblock[0]*nblock[1]) == 0:
    #         plt.imshow(block[:,:,0],cmap='gray')
    #         plt.show()
    #         print('Saving to binary file')
    #         block = np.swapaxes(block, 0, 1)
    #         block = np.swapaxes(block, 0, 2)
    #         ru.save_to_raw(saveto, block)
    #         block = np.zeros((dims[0], dims[1], subdims[2]))
    #         row = 0
    #         col = 0
        
        