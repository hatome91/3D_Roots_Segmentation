#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:34:50 2020

@author: fatome
"""
import roots_utils as ru
from datetime import datetime as dt

######################
# Global variable
######################

pathToFile = '/home/fatome/Downloads/Roots_1000x1000x1080.raw'
dims = (1000,1000,1080)
subdims = (100,100,108)




######################
# Main 
######################
if __name__ == '__main__':
    
    volGen = ru.load_region(pathToFile, dims, subdims)
    
    print(help(ru.load_region))