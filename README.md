# 3D_Roots_Segmentation
-----------------
DATA DESCRIPTION
-----------------

Sample 3D volume of roots can be found under the link:

https://drive.google.com/file/d/1N3mpX0oJlffgAo67K7ReKoj_zfwizMu7/view?usp=sharing

The tomogram can be visualized in imageJ:

- image type: 16-bit Unsigned
- width: 1000 pixels
- height: 1000 pixels
- offset: 0 bytes
- num images: 1080
- gap: 0
- with little-endian byte order

--------------------
MODULES DESCRIPTION
--------------------
* roots_utils.py
  - load_region(): genetor that yields region by region the 3D volume.
  - load_slice(): generator that yields slice by slice the 3D volume.
  - preprocessing(): function that could perform on demand the following operations:
    + my_clahe(): contrast limited adaptive histogram equalization
    + bilfil_thresh(): bilateral filter followed by Otsu thresholding
    + edmfil_thresh(): euclidian distance transformation followed by Otsu thresholding
  - convert_dtype(): function that converts among the following data types: uint8, uint16, float16, float32, int32
  - check_bilateral(): function that can be use to check the influence of the sigmaR and sigmaS parameters of the bilateral filter.
* TO BE CONTINUED
