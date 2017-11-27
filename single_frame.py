# imports, don't touch them lol
import cv2
import os
import numpy as np
import functions as f
import stereovision as sv

# ---------------------------------------------------------------------------

# obvious variable name for the dataset directory
dataset_path = "TTBB-durham-02-10-17-sub10";

# optional edits (if needed)
directory_to_cycle_left = "left-images";
directory_to_cycle_right = "right-images";

# set to timestamp to skip forward to, optional (empty for start)
# e.g. set to 1506943191.487683 for the end of the Bailey, just as the vehicle turns
skip_forward_file_pattern = "";

# ---------------------------------------------------------------------------

# resolve full directory location of data set for left / right images
path_dir_l =  os.path.join(dataset_path, directory_to_cycle_left);
path_dir_r =  os.path.join(dataset_path, directory_to_cycle_right);

# get a list of the left image files and sort them (by timestamp in filename)
filelist_l = sorted(os.listdir(path_dir_l));


# load a single file.
filename_l = "1506942475.481834_L.png"

options = {
    'crop_disparity' : False,       # display full or cropped disparity image
    'pause_playback' : False,       # pause until key press after each image
    'max_disparity' : 128,
    'ransac_trials' : 600,
    'road_color_thresh': 10,        # remove points from roadpts if it isn't in the x most populous colours 
    'point_threshold' : 0.05,
    'image_tiles' : True,           # show all images involved in the process or not
    'img_size' : (544,1024),
    'threshold_option' : 'previous', # options are: 'previous' or 'mean'
    'loop': False,
    'record_video' : False,
    'record_stats' : False,
    'video_filename' : 'previous.avi'
}

# from the left image filename get the corresponding right image
imageStores = f.loadImages(filename_l, path_dir_l, path_dir_r)
if imageStores != False:
    # load left and right image channels.
    imgL, imgR = imageStores
    # perform stereo vision!
    imgL, _ = sv.performStereoVision(imgL, imgR, None, options)
    # display results.
    cv2.imshow('left image',imgL)
    cv2.waitKey(0);
else:
    # looks like there was an error in loading the images.
    print("-- files skipped (perhaps one is missing or not PNG)");

# close all windows
cv2.destroyAllWindows()