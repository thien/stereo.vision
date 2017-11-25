
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

# check to handle video in the event that the user has requested it in options.
if options['record_video']:
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # cv2.VideoWriter_fourcc() does not exist
    video_writer = cv2.VideoWriter(options['video_filename'], fourcc, 8, (1024, 272))
else:
    fourcc = None
    video_writer = None

# disparity placeholder (for the next loop)
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
        # load image files
        imgL, imgR = f.loadImages(imgPaths)
        # compute stereo vision
        image, previousDisparity, normal = sv.performStereoVision(imgL, imgR, previousDisparity, options)

        if options['record_video']:
            video_writer.write(image)

        # print filenames and normals.
        f.printFilenamesAndNormals(filename_l, normal)

    else:
        print("-- files skipped (perhaps one is missing or not PNG)")

if options['record_video']:
    print("Video saved to:", options['video_filename'])
    video_writer.release()

# close all windows
cv2.destroyAllWindows()
