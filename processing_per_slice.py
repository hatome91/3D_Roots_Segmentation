#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 16:20:55 2020

@author: fatome
"""

import roots_utils as ru
from datetime import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from skimage.filters import threshold_multiotsu, sobel
from skimage.morphology import disk, binary_dilation
from skimage.measure import label, regionprops_table
import pandas as pd
from skimage.color import label2rgb


######################
# Global variable
######################

pathToFile = '/home/fatome/GitHub_Repos/3D_Roots_Segmentation/Tomo/Beans_Small_Real_Soil.raw'
pathToMask = '/home/fatome/GitHub_Repos/3D_Roots_Segmentation/mask.png'
dims = (1610,1610,1376)
subdims = (115,115,86)
nblock = np.divide(dims, subdims).astype(np.int)
saveto = 'Tomo/Real_Soil_thresh_Slices.raw'


t0=dt.now()


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
img = cv2.imread(pathToFile, 0) * mask


# plt.imshow(img,cmap='gray')
# plt.show()

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
imgA = clahe.apply(img)
# plt.imshow(imgA,cmap='gray')
# plt.imsave('2_Clahe.png', imgA, cmap='gray')
# plt.show()

imgB = ru.bilfil_thresh(imgA, 0.4, 3)
# plt.imshow(imgB,cmap='gray')
# plt.imsave('3_bilfil.png', imgB, cmap='gray')
# plt.show()

imgC = sobel(imgB, mask=mask.astype(bool))
# plt.imsave('4_Sobel.png', imgC, cmap='gray')
imgC_Inv = np.abs(imgC - np.max(imgC))
# plt.imsave('5_SobelInv.png', imgC_Inv, cmap='gray')

imgD = np.zeros_like(imgC_Inv)
imgD[imgC_Inv > 0.9 * np.max(imgC_Inv)] = 1
# plt.imsave('6_SobelInvThresh.png', imgD, cmap='gray')

# thr = threshold_multiotsu(imgB[300:900, 300:900], nbins=300)
thr = [0.27781035, 0.49368328]
imgE = np.zeros_like(imgB)
imgE[imgB>thr[1] * 0.9] = 1
# plt.imsave('7_Threshold.png', imgE, cmap='gray')

kernel = disk(3)
imgF = binary_dilation(imgE, selem=kernel)
# plt.imsave('8_dilate.png', imgF, cmap='gray')
imgG = imgF * imgD
# plt.imsave('9_dilateProduct.png', imgG, cmap='gray')

imgH = label(imgG, connectivity = 2)
props = pd.DataFrame(regionprops_table(imgH,img, properties=('label', 'area')))

imgI = np.zeros_like(img)

area = props[props['area']>150]

for lab in area['label']:
    imgI[imgH == lab] = lab

# imgI = label2rgb(imgI, img)

# plt.imsave('10_Labeled.png', imgI)


print(dt.now()-t0)