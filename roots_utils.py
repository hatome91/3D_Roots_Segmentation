import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import cv2

from skimage.util.shape import view_as_blocks
from skimage.segmentation import find_boundaries, watershed
from skimage.measure import regionprops_table, label
from skimage.util import invert
from skimage.filters import threshold_multiotsu, sobel
from skimage.morphology import disk, binary_dilation
from skimage.color import label2rgb


def load_region(filename, dims=(1000, 1000, 1080), block_dims=(100,100,108)):
    """        
        Parameters
        ----------
            filename : string
                Path to the raw file containing the data (3D volume) to be 
                loaded.
            dims : tuple, optional
                Tuple with the dimensions of the 3D volume. The default is 
                (1000, 1000, 1080).
            block_dims : tuple, optional
                Tuple with the dimensions of the small regions that will be 
                yieled. The default is (100,100,108).
            
        Yields
        ------
            numpy array uint16
                3D volume with shape block_dims.
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
    """
    

    Parameters
    ----------
        filename : string
            Path to the raw file containing the data (3D volume) to be loaded.
        dims : tuple, optional
            Tuple with the dimensions of the 3D volume. The default is 
            (1000, 1000, 1080).

    Yields
    ------
        numpy array uint16
            Slice of the 3D volume with shape (dims[0], dims[1])

    """
    
    with open(filename, 'rb') as f:
        
        for z in range(dims[2]):
            img_array = np.fromfile(f, dtype = np.uint16, count=dims[0]*dims[1])
            yield img_array.reshape(dims[0], dims[1])






def preprocessing(vol, CLAHE=True, BILFIL=(True, 0.4, 3, True), EDMFIL=(True, 0.3, True)):
    """
    

    Parameters
    ----------
    vol : numpy array
        Three dimensional volume to be processed.
    CLAHE : boolean, optional
        If True, then a Contrast Limited Adaptive Histogram Equalization 
        operation is performed. The default is True.
    BILFIL : tuple, optional
        The first component of the tuple is a boolean which indicates wheather 
        a bilateral filter will be applied or not.  If True, then a bilateral 
        filter is applied with the values given in the second and third 
        components of the tuple. The 2nd and 3rd components represent the 
        sigma_range and sigma_spatial of the bilateral filter respectively. 
        Finally the region is thresholded using Otsu thresholding, if the 4th 
        component is True. The default is (True, 0.4, 3, True).
    EDMFIL : tuple, optional
        The first component of the tuple is a boolean which indicates wheather
        an Euclidian Distance Map (EDM) transformation is applied or not. The 
        second component controls the thresholding level of the obtained EDM.
        This thresholding is applied if the 3rd component is True. The default
        is (True, 0.3, True).

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

def bilfil_thresh(src, sigmaR, sigmaS, applyThresh = True):
    
    if src.dtype != 'float32':
        src = convert_dtype(src,'float32')
        
    # Applying bilateral filter (only works with uint8 or float32)
    src = cv2.bilateralFilter(src, -1, sigmaR, sigmaS)
    
    # Convert to uint8
    if applyThresh:
        if src.dtype != 'uint8':
            src = convert_dtype(src, 'uint8')
        
        # Otsu Thresholding (only works with uint8)
        _, src = cv2.threshold(src, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return src

def edmfil_thresh(src, thr, applyThresh = True):
    
    if src.dtype != 'uint8':
        src = convert_dtype(src, 'uint8')
            
    # Euclidian distance map (works only with uint8)
    src = cv2.distanceTransform(src, cv2.DIST_L2, maskSize=5)
        
    # Normalize the edm for the range (0,1)
    cv2.normalize(src, src, 0.0, 1.0, cv2.NORM_MINMAX)
        
    #Thresholding the edm
    if applyThresh:
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
    BEFORE PROCESSING THE ENTIRE VOLUME IT IS RECOMMENDED TO OBTAIN THE OPTIMAL
    PARAMETERS FOR APPLYING BILATERAL FILTER, I.E., SIGMA_R AND SIGMA_S. THIS
    FUNCTION COMPUTES DIFFERENT COMBINATIONS OF SUCH PARAMETERS

    Parameters
    ----------
    src : numpy array. 
        Image for applying the bilateral filter.
    sigmaR : array, optional
        Values for checking the sigma_range part of the bilateral filter. The 
        default is None.
    sigmaS : array, optional
        Values for checking the sigma_spatial part of the bilater filter. The 
        default is None.
    fname : string, optional
        Path contating a name for saving the image with different sigmaR and 
        sigmaS. If the path does not contains an extension, png is 
        automatically assigned. The default is None.

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


def save_to_raw(saveto, vol):
    """
    THIS FUNCTION SAVE THE INFORMATION CONTAINED IN vol TO A BINARY RAW FILE

    Parameters
    ----------
    saveto : string
        Full path pointing to the file on which vol will be save.
    vol : numpy array
        Data to be save, it should be a 2D or 3D numpy array.

    Returns
    -------
    None.

    """
    if len(vol.shape) == 3:
        vol = np.swapaxes(vol, 0, 1)
        vol = np.swapaxes(vol, 0, 2)
    
    with open(saveto,'ab') as f:
        f.write(vol.astype('H').tostring())


def labelling3D(thresh):
    
    lab = find_boundaries(thresh, connectivity = thresh.ndim, mode = 'inner', background = 0)
    return watershed(lab, mask=thresh.astype(bool))



def check_threshold(pathToFile, dims, minArea=150, Slice=None, sigmaR=0.4, sigmaS=3, pathToMask=None, edgeThreshFactor=0.9, imgThreshFactor=0.9):
    """
    BEFORE STARING PROCESSING THE ENTIRE VOLUME IT IS RECOMMENDED TO CHECK ON A
    TEST IMAGE THAT THE LABELLING IS CORRECTLY DONE. THIS FUNCTIONS CARRIES OUT
    SUCH CHECKING AND SAVE ALL THE INTERMEDIATE IMAGES.

    Parameters
    ----------
    pathToFile : str
        Path to the raw file containing the data (3D volume) to be loaded..
    dims : tuple
        Tuple with the dimensions of the 3D volume. The default is 
        (1000, 1000, 1080).
    minArea : int
        Integer defining the minimum area that the detected regions within the 
        image should have, not to be considered noise. The default is 150.
    Slice : int, optional
        Integer defning a specific slice of the 3D volume to be analised and 
        obtain the optimal threshold values for binirising. If not number is 
        given, a slice in the middle of the volume is analised.
    sigmaR : TYPE, optional
        Sigma_range part of the bilater filter. The default is 0.4.
    sigmaS : int, optional
        Sigma_spatial part of the bilater filter. The default is 3.
    pathToMask : str, optional
        Path to a black and white image, enclosing in white the region of 
        interest. The default is None.
    edgeThreshFactor : int, optional
        It controls how much the edges found using sobel can penetrate the 
        detected regions. The default is 0.9.
    imgThreshFactor : int, optional
        The optimal thresholds can be incresed/decreased by this factor. 
        Sometimes this helps to better account for smaller fine details. The 
        default is 0.9.

    Returns
    -------
    thr : tuple
        Threshold values for binarising the image for the len(thr) classes 
        found.

    """
        
    if Slice == None:
        Slice = int(dims[2] /2)
    
    imgGen = load_slice(pathToFile, dims)
    
    
            
    for n, img in enumerate(imgGen):
        
        if n % Slice == 0 and n != 0:
            
            # Load the mask and use it to select the region of interest within the image
            if pathToMask != None:
                
                mask = cv2.imread(pathToMask, 0)
                mask = (np.array(mask) / 255).astype(np.uint8)
                img = img * mask
            else:
                mask = np.ones_like(img)
            
            plt.imsave('00_Slice_' + str(n) + '.png', img, cmap='gray')
            
            thr, imgC, imgB = get_multiotsu_thresholds(img, sigmaR, sigmaS, False)
            
            # Contrast Limited Adaptive Histogram Equalization
            plt.imsave('01_CLAHE_' + str(n) + '.png', imgC, cmap='gray')
            
            # Bilateral filter
            plt.imsave('02_BilFilt_' + str(n) + '.png', imgB, cmap='gray')
            
            # Edge detection
            imgEdges = sobel(imgB, mask = mask.astype(bool))
            plt.imsave('03_Sobel_' + str(n) + '.png', imgEdges, cmap='gray')
            imgEdges_Inv = invert(imgEdges)
            plt.imsave('04_Sobel_Iverted_' + str(n) + '.png', imgEdges_Inv, cmap='gray')
            
            # Binarising and setting the edges to black and the background to white
            imgBlackEdges = np.zeros_like(img)
            imgBlackEdges[imgEdges_Inv > edgeThreshFactor * np.max(imgEdges_Inv)] = 1
            plt.imsave('05_Binary_Black_Edges_' + str(n) + '.png', imgBlackEdges, cmap='gray')
            
            
            # Thresholding the image for the highest class
            imgThresh = np.zeros_like(imgB)
            imgThresh[imgB > thr[-1] * imgThreshFactor] = 1
            plt.imsave('06_Thresholded_Image_' + str(n) + '.png', imgThresh, cmap='gray')
            
            # Dilating the image
            kernel = disk(3)
            imgThresh = binary_dilation(imgThresh, selem = kernel)
            plt.imsave('07_Dilated_' + str(n) + '.png', imgThresh, cmap='gray')
            
            # Combining the information from the dilate and the edges images
            out = imgThresh * imgBlackEdges
            plt.imsave('08_Disconnected_' + str(n) + '.png', out, cmap='gray')
            outLabel = label(out, connectivity = 2)
            plt.imsave('09_Disconnected_Labelled_' + str(n) + '.png', outLabel, cmap='gray')
            
            # Obtaining the properties of the found regions/objects
            props = pd.DataFrame(regionprops_table(outLabel, properties = ('label', 'area')))
            
            # Regions with area smaller than the predifined minArea are not considered
            area = props[props['area'] > minArea]
            
            # Generating the image with the valid regions
            outLabel_Valid = np.zeros_like(img)
            for lab in area['label']:
                outLabel_Valid[outLabel == lab] = lab
            
            outLabel_Valid = label2rgb(outLabel_Valid, imgB, bg_label=-1)
            plt.imsave('10_Labelled_Valid_' + str(n) + '.png', outLabel_Valid)
            
            return thr
    
    
def get_multiotsu_thresholds(img, sigmaR=0.4, sigmaS=3, threshbilfil=False):
    """
    
    BEFORE STARING PROCESSING THE ENTIRE VOLUME IT IS RECOMMENDED TO OBTAIN THE
    OPTIMAL PARAMETERS FOR BINIRISING THE IMAGE. THIS FUNCTION, RETURNS THE 
    OPTIMAL THRESHOLD FOR THE DIFFERENT DETECTED CLASSES WITHIN A TEST 
    IMAGE/SLICE

    Parameters
    ----------
    img : 2D numpy array
        Representative image from where the thresholds will be computed.
    sigmaR : TYPE, optional
        Sigma_range part of the bilater filter. The default is 0.4.
    sigmaS : int, optional
        Sigma_spatial part of the bilater filter. The default is 3.

    Returns
    -------
    thr : tuple
        Threshold values for binarising the image for the len(thr) classes 
        found.
    imgC : 2D numpy array
        Image after Contrast Limited Adaptive Histogram Equalization.
    imgB : 2D numpy array
        Image after Contrast Limited Adaptive Histogram Equalization and 
        Bilateral filter.

    """
    
    # Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit = 3.0, tileGridSize = (8, 8))
    imgC = clahe.apply(img)
        
    # Bilateral filter
    imgB = bilfil_thresh(imgC, sigmaR, sigmaS, False)
    
    dims = imgB.shape
    # Computing the threshold for image thresholding via multi-otsu
    thr = threshold_multiotsu(imgB[int(dims[1] / 3):int(2 * dims[1] / 3), int(dims[0] / 3):int(2 * dims[0] / 3)], nbins=300)
     
    return thr, imgC, imgB



def img_processing(img, thr, mask, minArea=150, edgeThreshFactor=0.9, imgThreshFactor=0.9):
    """
    THIS FUNCTION APPLIES THE FOLLOWING OPERATIONS IN ORDER TO SEGMENT AND 
    IMAGE:
        - CLAHE
        - BILATERAL FILTER
        - SOBEL EDGE DETECTION
        - INVERTS OUTPUT FROM SOBEL EDGE DETECTION
        - BINARISES INVERTED SOBEL
        - BINARISES THE BILATERAL FILTERED IMAGE
        - DILATES THE BINARISED IMAGE
        - MULTIPLIES THE DILATED IMAGE WITH THE BINARISED INVERTED SOBEL IMAGE
        - LABELS THE PREVIOUS IMAGE
        - OBTAINS THE PROPERTIES OF THE LABEL REGIONS
        - DROPS REGIONS WITH SMALLER AREAN THAN A PREDIFINED THRESHOLD
        - CREATES A BINARY IMAGE CONTAINING THE REMAINING VALID REGIONS

    Parameters
    ----------
    img : 2D numpy array
        Image to be analised.
    thr : int
        Threshold for binirising the image with respect to a certain class.
    mask : 2D numpy array
        Black and white image, enclosing in white the region of interest.
    minArea : int, optional
        Integer defining the minimum area that the detected regions within the 
        image should have, not to be considered noise. The default is 150.
    edgeThreshFactor : int, optional
        It controls how much the edges found using sobel can penetrate the 
        detected regions. The default is 0.9.
    imgThreshFactor : int, optional
        The optimal thresholds can be incresed/decreased by this factor. 
        Sometimes this helps to better account for smaller fine details. The 
        default is 0.9.

    Returns
    -------
    out_Valid : 2D numpy array
        Thresholded binary image containing just valid regions, according to 
        the minArea criteria.

    """
    img = img * mask
    
    # Edge detection
    imgEdges = sobel(img, mask = mask.astype(bool))
    imgEdges_Inv = invert(imgEdges)
        
    # Binarising and setting the edges to black and the background to white
    imgBlackEdges = np.zeros_like(img)
    imgBlackEdges[imgEdges_Inv > edgeThreshFactor * np.max(imgEdges_Inv)] = 1
        
       
    # Thresholding the image for the highest class
    imgThresh = np.zeros_like(img)
    imgThresh[img>thr[-1] * imgThreshFactor] = 1
        
    # Dilating the image
    kernel = disk(3)
    imgThresh = binary_dilation(imgThresh, selem = kernel)
        
    # Combining the information from the dilate and the edges images
    out = imgThresh * imgBlackEdges
    outLabel = label(out, connectivity = 2)
        
    # Obtaining the properties of the found regions/objects
    props = pd.DataFrame(regionprops_table(outLabel, properties = ('label', 'area')))
        
    # Regions with area smaller than the predifined minArea are not considered
    area = props[props['area']>minArea]
        
    # Generating the image with the valid regions
    out_Valid = np.zeros_like(img)
    for lab in area['label']:
        out_Valid[outLabel == lab] = 1
    
    return out_Valid

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    