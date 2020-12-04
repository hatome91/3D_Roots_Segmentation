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
  - preprocessing(): function that perform the following operations:
    + contrast limited adaptive histogram equalization
    + 
