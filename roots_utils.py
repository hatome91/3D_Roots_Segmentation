import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.util.shape import view_as_blocks
from skimage.segmentation import find_boundaries, watershed
import pandas as pd
from skimage.measure import regionprops_table


def load_region(filename, dims=(1000, 1000, 1080), block_dims=(100,100,108)):
    """        
        Parameters
        ----------
            filename : string
                Path to the raw file containing the data (3D volume) to be loaded.
            dims : tuple, optional
                Tuple with the dimensions of the 3D volume. The default is (1000, 1000, 1080).
            block_dims : tuple, optional
                Tuple with the dimensions of the small regions that will be yieled. The default is (100,100,108).
            
        Yields
        ------
            numpy array uint16
                3D volume with dimensions block_dims.
    """

    with open(filename, 'rb') as f:
        
        # Loops over the amount of blocks in the z direction
        for i in range(0,int(dims[2]/block_dims[2])):
            
            # Initialize the volume as a numpy array
            vol = np.zeros((dims[0], dims[1], block_dims[2]))
            
            # Loops over the amount of slices in the z direction
            for b in range(block_dims[2]):
                
                # Loads slices with dimensions dims[0] x dims[1]
                img_array = np.fromfile(f, dtype = np.uint16, count=dims[0]*dims[1])
                img = img_array.reshape(dims[0], dims[1])
                
                # Updates the volume with the new slice
                vol[:,:,b] = img
            
            # Divides the volume into blocks with dimension given by block_dims
            blocks = view_as_blocks(np.array(vol), block_shape = block_dims)
            
            # Loops over the amount of blocks in the x direction
            for b1 in range(blocks.shape[0]):
                
                # Loops over the amount of blocks in the y direction
                for b2 in range(blocks.shape[1]):
                    
                    # Yields a numpy array with dimensions given by block_dims
                    yield np.array(blocks[b1, b2,0,:,:], dtype=np.uint16).reshape(block_dims)
        


def load_slice(filename, dims=(1000,1000,1080)):
    
    with open(filename, 'rb') as f:
        
        for z in range(dims[2]):
            img_array = np.fromfile(f, dtype = np.uint16, count=dims[0]*dims[1])
            img = img_array.reshape(dims[0], dims[1])
            yield img






def preprocessing(vol, CLAHE=True, BILFIL=(True, 0.4, 3), EDMFIL=(True, 0.3)):
    """
    

    Parameters
    ----------
    vol : numpy array
        Three dimensional volume to be processed.
    CLAHE : boolean, optional
        If True, then a Contrast Limited Adaptive Histogram Equalization operation is performed. The default is True.
    BILFIL : tuple, optional
        The first component of the tuple is a boolean which indicates wheather a bilateral filter will be applied or not. 
        If True, then a bilateral filter is applied with the values given in the second and third components of the tuple.
        The 2nd and 3rd components represent the sigma_range and sigma_spatial of the bilateral filter respectively. 
        Finally the region is thresholded using Otsu thresholding. The default is (True, 0.4, 3).
    EDMFIL : tuple, optional
        The first component of the tuple is a boolean which indicates wheather an Euclidian Distance Map (EDM) transformation is applied or not.
        The second component controls the thresholding level of the obtained EDM. The default is (True, 0.3).

    Returns
    -------
    out : numpy array
       Thresholded three dimensional volume.

    """
    
    # Check the dimensions of the input
    if vol.ndim == 3:
    
        # Retrieve the dimensions
        xdim, ydim, zdim = vol.shape
    
        out = np.zeros_like(vol,dtype=np.float32)
    
        # Loop over the different slices (z direction)
        for ind in range(zdim):
        
            # Extract a slice
            src = vol[:,:,ind]
            
            if CLAHE:
            
                src = my_clahe(src)
        
            if BILFIL[0]:
            
                src = bilfil_thresh(src, BILFIL[1], BILFIL[2])
            
            if EDMFIL[0]:
            
                src = edmfil_thresh(src, EDMFIL[1])
        
            # Convert to float32
            if src.dtype != 'float32':
                out[:,:,ind] = convert_dtype(src, 'float32')
            else:
                out[:,:,ind] = src
        
        return out
    
    elif vol.ndim == 2:
        
        # Retrieve the dimensions
        xdim, ydim = vol.shape
                    
        if CLAHE:
            
            vol = my_clahe(vol)
        
        if BILFIL[0]:
            
            vol = bilfil_thresh(vol, BILFIL[1], BILFIL[2])
            
        if EDMFIL[0]:
            
            vol = edmfil_thresh(vol, EDMFIL[1])
        
        # Convert to float32
        if vol.dtype != 'float32':
            return convert_dtype(vol, 'float32')
        
        return vol

def my_clahe(src):
    
    # Convert to uint8
    if src.dtype != 'uint8' or src.dtype != 'uint16':
        src = convert_dtype(src,'uint8')
    
    # Applying Contrast Limited Adaptative Histogram Equalization (works with uint8 or uint16)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    src = clahe.apply(src)
                   
    return src

def bilfil_thresh(src, sigmaR, sigmaS):
    
    if src.dtype != 'uint8' or src.dtype != 'float32':
        src = convert_dtype(src,'float32')
        
    # Applying bilateral filter (only works with uint8 or float32)
    src = cv2.bilateralFilter(src, -1, sigmaR, sigmaS)
    
    # Convert to uint8
    if src.dtype != 'uint8':
        src = convert_dtype(src, 'uint8')
        
    # Otsu Thresholding (only works with uint8)
    _, src = cv2.threshold(src, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return src

def edmfil_thresh(src, thr):
    
    if src.dtype != 'uint8':
        src = convert_dtype(src, 'uint8')
            
    # Euclidian distance map (works only with uint8)
    src = cv2.distanceTransform(src, cv2.DIST_L2, maskSize=5)
        
    # Normalize the edm for the range (0,1)
    cv2.normalize(src, src, 0.0, 1.0, cv2.NORM_MINMAX)
        
    #Thresholding the edm
    _, src = cv2.threshold(src, thr, 1.0, cv2.THRESH_BINARY)
    return src

def labelling_per_layer(threshs):
    
    # Retrieve the dimensions
    _, _, zdim = threshs.shape
    
    labelled = np.zeros_like(threshs)
    
    # Loop over all the slices in the z-direction
    for ind in range(zdim):
        
        # Extract a slice
        thresh = threshs[:,:,ind] 
        
        # Find markers
        thresh = convert_dtype(thresh, 'uint8')
        _, markers = cv2.connectedComponents(thresh)
        #contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        markers += 1
        
        # Create the marker image
        #markers = np.zeros_like(thresh, dtype=np.uint8)
        
        # for i in range(len(contours)):
        #     markers = cv2.drawContours(markers, contours, i, i+1, -1)
                
        # c = np.where(thresh==0)
        
        # markers[c[0][:],c[1][:]]=255
        
        # Convert the markers to uint32 and the thresh to RGB
        markers = convert_dtype(markers, 'int32')
        thresh = convert_dtype(cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB), 'uint8')
        
        # # Segment the image using watershed
        # markers = cv2.watershed(thresh, markers)
        
        # # Get the properties of the segmented regions
        # prop = pd.DataFrame(regionprops_table(markers,properties=('label','area')))
        
        # # Eliminate small regions
        # areaMin = 20
        # a = prop[prop['area']<areaMin]
        
        # for i in a['label']:
        #     ind = np.where(markers == i)
        #     markers[ind[0][:],ind[1][:]]=0
        
        labelled[:,:,ind] = markers
        
    labelled = convert_dtype(labelled, 'uint16')
    return labelled











def convert_dtype(img, dType):
    """
    

    Parameters
    ----------
    img : numpy array
        Array which data type will be change to dType.
    dType : string
        String representing the new data type of the array. 
        Conversion are possible among the following data types:
            'uint8', 'uint16', 'int32' and 'float32'

    Returns
    -------
    numpy array
        Array of the same dimensions as img with data type is given by dType.

    """
    
    if dType == 'uint8':
        if img.dtype != 'float32' and img.dtype != 'float16':
            return np.array(img/np.iinfo(img.dtype).max * np.iinfo(np.uint8).max,dtype=np.uint8)
        else:
            n = img-np.min(img)
            d = np.max(n)
            return np.array(n / d * np.iinfo(np.uint8).max,dtype=np.uint8)
    
    elif dType == 'uint16':
        if img.dtype != 'float32' and img.dtype != 'float16':
            return np.array(img/np.iinfo(img.dtype).max * np.iinfo(np.uint16).max, dtype=np.uint16)
        else:
            n = img-np.min(img)
            d = np.max(n)
            return np.array(n / d * np.iinfo(np.uint16).max, dtype=np.uint16)
    
    elif dType == 'float32':
        return np.array(img/np.iinfo(img.dtype).max, dtype=np.float32)
        
    elif dType == 'int32':
        if img.dtype != 'float32' and img.dtype != 'float16':
            return np.array(img/np.iinfo(img.dtype).max * np.iinfo(np.int32).max, dtype=np.int32)
        else:
            n = img-np.min(img)
            d = np.max(n)
            return np.array(n/d * np.iinfo(np.int32).max, dtype=np.int32)












def check_bilateral(src, sigmaR = None, sigmaS = None, fname=None):
    """
    

    Parameters
    ----------
    src : numpy array. 
        Image for applying the bilateral filter.
    sigmaR : array, optional
        Values for checking the sigma_range part of the bilateral filter. The default is None.
    sigmaS : array, optional
        Values for checking the sigma_spatial part of the bilater filter. The default is None.
    fname : string, optional
        Path contating a name for saving the image with different sigmaR and sigmaS. If the path does not contains an extension, png is automatically assigned. The default is None.

    Returns
    -------
    None.

    """
    
    # Checking the right parameters for the bilateral filter (works with uint8 or float32)
    print('Checking parameters for bilateral filter')
    
    if src.dtype != 'uint8' or src.dtype != 'float32':
        src = convert_dtype(src, 'float32')
    
    if sigmaR == None:
        sigmaR = np.linspace(10,100,10)/100
    if sigmaS == None:
        sigmaS = range(1,11,2)
    c=1
    l_SigmaR = len(sigmaR)
    l_SigmaS = len(sigmaS)
    
    plt.figure(figsize=(int(2*l_SigmaR),int(2*l_SigmaS)))
    for i in sigmaS:
        for j in sigmaR:
            plt.subplot(l_SigmaS, l_SigmaR, c)
            plt.imshow(cv2.bilateralFilter(src, -1, j, i), cmap='gray')
            plt.title(f's_r={j}, s_s={i}')
            c += 1
    
    if fname == None:
        plt.savefig('Bilateral_Parameters_Check.png')
    else:
        sp = fname.split('.')
        if len(sp) != 2:
            plt.savefig(fname[0] + '.png')
        else:
            plt.savefig(fname)











def save_to_raw():
    pass












def labelling3D(thresh):
    
    lab = find_boundaries(thresh, connectivity = thresh.ndim, mode = 'inner', background = 0)
    return watershed(lab, mask=thresh.astype(bool))

