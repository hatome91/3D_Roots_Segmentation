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


######################
# Global variable
######################

pathToFile = '/home/fatome/Downloads/Roots_1000x1000x1080.raw'
dims = (1000,1000,1080)
subdims = (100,100,108)
nblock = np.divide(dims, subdims).astype(np.int8)
saveto = 'thresh.raw'


###################################
# Check if the outputfile exists
###################################
if os.path.exists(saveto):
    os.remove(saveto)


######################
# Main 
######################
if __name__ == '__main__':
    
    imgGen = ru.load_slice(pathToFile, dims)
    
    for i, img in enumerate(imgGen):
        continue
    print(i+1)
    
    
    # volGen = ru.load_region(pathToFile, dims, subdims)
    
    # block = np.zeros((dims[0], dims[1], subdims[2]))
    
    # row = 0
    # col = 0
    
    # for n, vol in enumerate(volGen):
        
    #     thresh = ru.preprocessing(vol)
        
    #     if n % nblock[2] != 0 or col == 0:
    #         block[row*subdims[0]:(row+1)*subdims[0], col*subdims[1]:(col+1)*subdims[1], :] = thresh
    #         col += 1
    #     else:
    #         row += 1
    #         col = 0
    #         block[row*subdims[0]:(row+1)*subdims[0], col*subdims[1]:(col+1)*subdims[1], :] = thresh
    #         col += 1
        
    #     if (n + 1) % subdims[0] * subdims[1] == 0:
    #         block = np.swapaxes(block, 0, 1)
    #         block = np.swapaxes(block, 0, 2)
    #         with open(saveto,'ab') as f:
    #             f.write(block.astype('H').tostring())
    #         break
    #         block = np.zeros((dims[0], dims[1], subdims[2]))
    #         row = 0
    #         col = 0
        
        