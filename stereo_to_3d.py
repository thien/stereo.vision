#####################################################################

# Example : project SGBM disparity to 3D points for am example pair
# of rectified stereo images from a  directory structure
# of left-images / right-images with filesname DATE_TIME_STAMP_{L|R}.png

# basic illustrative python script for use with provided stereo datasets

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2017 Deparment of Computer Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################

import cv2
import os
import numpy as np
import random
import csv
import functions as f

master_path_to_dataset = "dataset" # ** need to edit this **
directory_to_cycle_left = "left-images"     # edit this if needed
directory_to_cycle_right = "right-images"   # edit this if needed


#####################################################################

# resolve full directory location of data set for left / right images

full_path_directory_left =  os.path.join(master_path_to_dataset, directory_to_cycle_left);
full_path_directory_right =  os.path.join(master_path_to_dataset, directory_to_cycle_right);

full_path_filename_left = os.path.join(full_path_directory_left, "1506942481.483936_L.png");
full_path_filename_right = (full_path_filename_left.replace("left", "right")).replace("_L", "_R");

# setup the disparity stereo processor to find a maximum of 128 disparity values
# (adjust parameters if needed - this will effect speed to processing)

max_disparity = 32;
stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21);

# for sanity print out these filenames

print(full_path_filename_left);
print(full_path_filename_right);

# check the files actually exist




def ransac(image):
    """
    Pseudocode:

    for i = 1 : trials:
        select T data points randomly
        estimate feature parameters using model
            check related colours
            check lines are close to others
        if number of data points > V
        return success
    return failure
    """



if (os.path.isfile(full_path_filename_left) and os.path.isfile(full_path_filename_right)) :

    # read left and right images and display in windows
    # N.B. despite one being grayscale both are in fact stored as 3-channel
    # RGB images so load both as such

    imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR);
    imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR);

    print("-- files loaded successfully");

    # remember to convert to grayscale (as the disparity matching works on grayscale)
    # N.B. need to do for both as both are 3-channel images

    grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY);
    grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY);

    # compute disparity image from undistorted and rectified stereo images
    # that we have loaded
    # (which for reasons best known to the OpenCV developers is returned scaled by 16)

    disparity = stereoProcessor.compute(grayL,grayR);

    # filter out noise and speckles (adjust parameters as needed)

    dispNoiseFilter = 5; # increase for more agressive filtering
    cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter);

    # scale the disparity to 8-bit for viewing
    # divide by 16 and convert to 8-bit image (then range of values should
    # be 0 -> max_disparity) but in fact is (-1 -> max_disparity - 1)
    # so we fix this also using a initial threshold between 0 and max_disparity
    # as disparity=-1 means no disparity available

    _, disparity = cv2.threshold(disparity,0, max_disparity * 16, cv2.THRESH_TOZERO);
    disparity_scaled = (disparity / 16.).astype(np.uint8);

    # display image (scaling it to the full 0->255 range based on the number
    # of disparities in use for the stereo part)
    some_disp =  (disparity_scaled * (255. / max_disparity)).astype(np.uint8)
    cv2.imshow("disparity", some_disp);

    # imgL = cv2.addWeighted(imgL,0.7,some_disp,0.3,0)
    # imgR = cv2.addWeighted(imgR,0.7,some_disp,0.3,0)
    # print(disparity.shape)
    # print(grayR.shape)
    # cv2.add(grayR, disparity)
    # grayR = cv2.addWeighted(grayR,0.7,disparity,0.3,0)
    # project to a 3D colour point cloud (with or without colour)

    # points = project_disparity_to_3d(disparity_scaled, max_disparity);
    points = f.project_disparity_to_3d(disparity_scaled, max_disparity, imgL);
    print(points[0])
    # write to file in an X simple ASCII X Y Z format that can be viewed in 3D
    # using the on-line viewer at http://lidarview.com/
    # (by uploading, selecting X Y Z format, press render , rotating the view)

    point_cloud_file = open('3d_points.txt', 'w');
    csv_writer = csv.writer(point_cloud_file, delimiter=' ');
    csv_writer.writerows(points);
    point_cloud_file.close();

    # select a random subset of the 3D points (4 in total)
    # and them project back to the 2D image (as an example)

    pts = f.project_3D_points_to_2D_image_points(random.sample(points, 4));
    pts = np.array(pts, np.int32);
    pts = pts.reshape((-1,1,2));

    cv2.polylines(imgL,[pts],True,(0,0,255), 3);

    cv2.imshow('left image',imgL)
    # cv2.imshow('right image',imgR)
    # cv2.imshow('left image',grayL)
    # cv2.imshow('right image',grayR)

    # wait for a key press to exit

    cv2.waitKey(0);

else:
        print("-- files skipped (perhaps one is missing or path is wrong)");
        print();

# close all windows

cv2.destroyAllWindows()

#####################################################################
