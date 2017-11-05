###########################################################################
#    THE KITTI VISION BENCHMARK: STEREO/FLOW/SCENE FLOW BENCHMARKS 2015   #
#                   Andreas Geiger         Moritz Menze                   #
#          Max Planck Institute for Intelligent Systems, Tübingen         #
#                       Leibniz Universität Hannover                      #
#                             www.cvlibs.net                              #
###########################################################################

This file describes the KITTI stereo 2015, flow 2015 and scene flow 2015
benchmarks, consisting of 200 training and 200 test image pairs for each 
task. Ground truth has been acquired by accumulating 3D point clouds from a 
360 degree Velodyne HDL-64 Laserscanner and fitting 3D CAD models to 
individually moving cars. Please have a look at our publications for details.

Dataset description:
====================

The training and testing folders contain the left and right images in the image_2
and image_3 subdirectories, respectively. Additionally, the training folder
contains the following subfolders containing ground truth for disparity and flow
(see the text below this section for a detailed description of the data format):

- disp_xxx_0: Disparity maps of first image pairs in reference frame
              (only regions which don't leave the image domain).

- disp_xxx_1: Disparity information of second image pair mapped into the
              reference frame (first left image) via the true optical flow.

- flow_xxx:   Optical flow which maps from the first left to the second left
              image. Specified in the reference frame (first left image).

'noc' refers to non-occluded regions, ie, regions for which the matching
correspondence is inside the image domain. 'occ' refers to all image
regions for which ground truth could be measured (including regions which map
to points outside the image domain in the other view). In the KITTI online
evaluation both types of image regions are evaluated (see detailed results of
the respective method), however for ranking the methods and for the main table
all image regions are considered (corresponding to the 'occ' folders).

Submission instructions:
========================

NOTE: WHEN SUBMITTING RESULTS, PLEASE STORE THEM IN THE SAME DATA FORMAT IN
WHICH THE GROUND TRUTH DATA IS PROVIDED (SEE BELOW), USING THE FILE NAMES
000000_10.png TO 000199_10.png. CREATE A ZIP ARCHIVE OF THEM AND STORE
YOUR RESULTS IN THE FOLLOWING DIRECTORIES LOCATED IN YOUR ZIP'S ROOT FOLDER:

- disp_0: Disparity maps of first image pair in reference frame
          (first left image); needed for the stereo and scene flow benchmark.
- disp_1: Disparity information of second image pair mapped into the reference (!)
          frame (first left image) via the optical flow; required only for
          submissions to the scene flow benchmark (for specifying the scene
          flow of every pixel in the reference frame, we specify the disparity in
          the first and second image and the optical flow, all represented in the
          reference frame! If your method represents the disparity estimation of
          the second image pair in the second left image, then you need to map
          it back to the first image and fill in the missing values).
- flow:   Optical flow which maps from the first left to the second left image,
          specified in the reference frame (first left image); needed for the
          optical flow and the scene flow benchmark.

Submission example:
===================

If you want to submit your scene flow results, your zip folder structure must
look like this (the png file format is specified later in this readme):

|-- zip
  |-- disp_0  (Disparity maps of first image pair)
    |-- 000000_10.png
    |-- ...
    |-- 000199_10.png
  |-- disp_1  (Disparity maps of second image pair, mapped to reference image)
    |-- 000000_10.png
    |-- ...
    |-- 000199_10.png
  |-- flow    (Flow fields between first and second image)
    |-- 000000_10.png
    |-- ...
    |-- 000199_10.png

If you want to submit stereo or flow results, only the folders "disp_0" or "flow"
must be present in the zip file, respectively.

File description:
=================

The folders testing and training contain the color video images in
the sub-folders image_2 (left image) and image_3 (right image). All input
images are saved as unsigned char color PNG images. Filenames are
composed of a 6-digit image index as well as a 2-digit frame number:

 - xxxxxx_yy.png

Here xxxxxx is running from 0 to 199 and the frame number yy is either 10 or 11. 
The reference images, for which all results must be provided, are the left images 
of frame 10 for each test pair.

Corresponding ground truth disparity maps and flow fields can be found in
the folders disp_0, disp_1 and flow of the training set, respectively. The suffix 
_noc or _occ refers to non-occluded or occluded (=all pixels).

File naming examples:
=====================

Test stereo pair '000005':

 - left image:  testing/image_2/000005_10.png
 - right image: testing/image_3/000005_10.png

Test flow pair '000005':

 - first frame:  testing/image_2/000005_10.png
 - second frame: testing/image_2/000005_11.png
 
Scene flow test scene '000005':

 - left image of the first pair:   testing/image_2/000005_10.png
 - right image of the first pair:  testing/image_3/000005_10.png
 - left image of the second pair:  testing/image_2/000005_11.png
 - right image of the second pair: testing/image_3/000005_11.png 
 
Data format:
============

Disparity and flow values range [0..256] and [-512..+512] respectively. For
both image types documented MATLAB and C++ I/O functions are provided
within this development kit in the folders matlab and cpp. If you want to
use your own code instead, you need to follow these guidelines:

Disparity maps are saved as uint16 PNG images, which can be opened with
either MATLAB or libpng++. A 0 value indicates an invalid pixel (ie, no
ground truth exists, or the estimation algorithm didn't produce an estimate
for that pixel). Otherwise, the disparity for a pixel can be computed by
converting the uint16 value to float and dividing it by 256.0:

disp(u,v)  = ((float)I(u,v))/256.0;
valid(u,v) = I(u,v)>0;

Optical flow maps are saved as 3-channel uint16 PNG images: The first channel
contains the u-component, the second channel the v-component and the third
channel denotes if the pixel is valid or not (1 if true, 0 otherwise). To convert
the u-/v-flow into floating point values, convert the value to float, subtract
2^15 and divide the result by 64.0:

flow_u(u,v) = ((float)I(u,v,1)-2^15)/64.0;
flow_v(u,v) = ((float)I(u,v,2)-2^15)/64.0;
valid(u,v)  = (bool)I(u,v,3);

The evaluation of scene flow results is based on 3D motion vectors. In addition to 
the disparity and optical flow maps of the reference frame it depends on disparity
estimates for the second time step which we represent in the reference frame as well
(ie, they are warped via the optical flow). This guarantees that for every pixel of
the reference frame a 3D scene flow vector can be stored via two disparity maps
(disp_0 and disp_1) and one flow map (flow). The actual 3D vector can be obtained
by accessing the calibration files, but our evaluation takes place in image space
(disparity and flow space) as all measurements have been obtained in image space.

Evaluation Code:
================

For transparency we have included the KITTI 2015 evaluation code in the
sub-folder 'cpp' of this development kit. It can be compiled via:

g++ -O3 -DNDEBUG -o evaluate_scene_flow evaluate_scene_flow.cpp -lpng

Mapping to Raw Data
===================

Note that this section is additional to the benchmark, and not required for
solving the benchmark tasks.

In order to allow the usage of the laser point clouds, gps data and the 
grayscale images for the TRAINING data as well, we provide the mapping of 
the training set to the raw data of the KITTI dataset.

This information is saved in mapping/train_mapping.txt:

train_mapping.txt: Each line (0-based numbering) maps an image ID to a 
zip file of the KITTI raw dataset files. Note that those files are split into
several categories on the website! Lines corresponding to images for which
the sequence is not included in the public raw data (e.g., due to incomplete
or noisy tracklet annotations) are left empty.

Example: Image 000010_10.png from the training set maps to date 2011_09_26, 
drive 9 and frame 384. Drives and frames are 0-based.

