"""
Task Specification – Road Surface Region Detection

    You are required develop to road surface detection system that correctly detects the 3D planar orientation and bounds (i.e. edges) of any free (unoccupied) road surface region immediately in- front of the vehicle in which the autonomous vehicle needs to operate (e.g. for staying in the correct lane / staying on the road itself / avoiding obstacles / automatic braking).

    In constructing your solution you may wish to consider the following aspects of your design:
    • exploring the optimization of the provided stereo vision algorithm in use and its operation and hows its performance under varying illumination conditions could perhaps be improved using the HSV or other colour spaces covered in CS SM L2 - Image Processing (these paper also presents an interesting research approach [1,2], others solutions exist also for illumination invariant colour spaces - use these as a starting point for your search).

    • selection of a region of interest, possibly adaptively, within the image (including possibly areas of road, pavement, other or not) that represents the region directly in-front of the vehicle and how to deal with the problem of missing disparity (depth) values in that region.

    • Calculating the equation of a plane from 3 points in 3D: http://mathworld.wolfram.com/Plane.html

    [Further hint: for this assignment this can be done in full projected floating-point 3D space (X,Y, Z)or in integer image space (x,y,disparity) – see provided hints python file]

    Your solution must use a RANdom SAmple and Consensus (RANSAC) approach to perform the detection of the 3D plane in front of the vehicle (when and where possible). For the avoidance of doubt, no credit will be given for a 2D solution based on the built-in Hough Transform or Douglas- Pecker contour detection in OpenCV or that does not recover the 3D parameters of the plane.

    Additionally, some example images may not have significant noise-free disparity (depth) available in front of the vehicle or the road region may be partially occluded by other objects (people, vehicles etc.). The road surface itself will change in terrain type, illumination conditions and road markings – ideally your solution should be able to cope with all of these. Road edges may or may not be marked by line markings in the colour image. All examples will contain a clear front facing view of the road in front of the vehicle only – your system should report all appropriate road surface plane instances it can detect recognising this may not be possible for all cases within the data set provided.
"""
"""
    http://resources.mpi-inf.mpg.de/TemporalStereo/articleJpeg.pdf
"""


# obvious variable name for the dataset directory
dataset_path = "TTBB-durham-02-10-17-sub10"

# optional edits (if needed)
directory_to_cycle_left = "left-images"
directory_to_cycle_right = "right-images"

# set to timestamp to skip forward to, optional (empty for start)
# e.g. set to 1506943191.487683 for the end of the Bailey, just as the vehicle turns
skip_forward_file_pattern = ""


options = {
    'crop_disparity' : False,       # display full or cropped disparity image
    'pause_playback' : False,       # pause until key press after each image
    'max_disparity' : 128,
    'ransac_trials' : 600,
    'road_color_thresh': 10,        # remove points from roadpts if it isn't in the x most populous colours 
    'loop': True,
    'point_threshold' : 0.05,
    'image_tiles' : True,           # show all images involved in the process or not
    'img_size' : (544,1024),
    'threshold_option' : 'previous', # options are: 'previous' or 'mean'
    'record_video' : True,
    'video_filename' : 'previous.avi'
}

# ------------------------------------------------------------------------
# Don't edit below this line!
# ------------------------------------------------------------------------

# imports, don't touch them lol
import cv2
import os
import random
import numpy as np
import csv
import functions as f
import stereovision as sv


# resolve full directory location of data set for left / right images
path_dir_l =  os.path.join(dataset_path, directory_to_cycle_left)
path_dir_r =  os.path.join(dataset_path, directory_to_cycle_right)

# get a list of the left image files and sort them (by timestamp in filename)
filelist_l = sorted(os.listdir(path_dir_l))


if options['record_video']:
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # cv2.VideoWriter_fourcc() does not exist
    video_writer = cv2.VideoWriter(options['video_filename'], fourcc, 8, (1024, 272))
else:
    fourcc = None
    video_writer = None

previousDisparity = None
for filename_l in filelist_l:
    """
    Here we'll cycle through the files, and finding each stereo pair.
    We'll then process them to detect the road surface planes, and compute 
    the stereo disparity.
    """

    # skip forward to start a file we specify by timestamp (if this is set)
    if ((len(skip_forward_file_pattern) > 0) and not(skip_forward_file_pattern in filename_l)):
        continue
    elif ((len(skip_forward_file_pattern) > 0) and (skip_forward_file_pattern in filename_l)):
        skip_forward_file_pattern = ""

    # get image paths
    imgPaths = f.getImagePaths(filename_l, path_dir_l, path_dir_r)
    if imgPaths != False:
        imgL, imgR = f.loadImages(imgPaths)

        image, previousDisparity, normal = sv.performStereoVision(imgL, imgR, previousDisparity, options)

        if options['record_video']:
            video_writer.write(image)

        filename_r = filename_l.replace("_L", "_R")
        print(filename_l);
        if normal is not None:
            print(filename_r + " - Road Surface Normal:" + f.NormalString(normal))
        else:
            print(filename_r + " - Road Surface Normal could not be computed")

    else:
        print("-- files skipped (perhaps one is missing or not PNG)")

if options['record_video']:
    print("Video saved to:", options['video_filename'])
    video_writer.release()

# close all windows
cv2.destroyAllWindows()
