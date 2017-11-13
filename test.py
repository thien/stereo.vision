# imports, don't touch them lol
import cv2
import os
import random
import numpy as np
import csv
import functions as f

# ---------------------------------------------------------------------------

# obvious variable name for the dataset directory
dataset_path = "dataset";

# optional edits (if needed)
directory_to_cycle_left = "left-images";
directory_to_cycle_right = "right-images";

# set to timestamp to skip forward to, optional (empty for start)
# e.g. set to 1506943191.487683 for the end of the Bailey, just as the vehicle turns
skip_forward_file_pattern = "";

crop_disparity = False;     # display full or cropped disparity image
pause_playback = False;     # pause until key press after each image

# ---------------------------------------------------------------------------

# resolve full directory location of data set for left / right images
path_dir_l =  os.path.join(dataset_path, directory_to_cycle_left);
path_dir_r =  os.path.join(dataset_path, directory_to_cycle_right);

# get a list of the left image files and sort them (by timestamp in filename)
filelist_l = sorted(os.listdir(path_dir_l));

# setup the disparity stereo processor to find a maximum of 128 disparity values
# (adjust parameters if needed - this will effect speed to processing)
max_disparity = 64;

# Start the loop
filename_l = "1506943132.486155_L.png"


# # from the left image filename get the correspondoning right image
imageStores = f.loadImages(filename_l, path_dir_l, path_dir_r)
if imageStores != False:
    imgL, imgR = imageStores

    imgL, imgR = f.preProcessImages(imgL,imgR)
    grayL, grayR = f.greyscale(imgL,imgR)
    # compute disparity image
    disparity = f.disparity(grayL,grayR, max_disparity, crop_disparity)
    disparity = f.maskDisparity(disparity)
    # show disparity
    cv2.imshow("disparity", disparity);

    # project to a 3D colour point cloud (with or without colour)
    points = f.projectDisparityTo3d(disparity, max_disparity, imgL);
    # print(points)
    # write to file in an X simple ASCII X Y Z format that can be viewed in 3D
    f.saveCoords(points, '3d_points.txt')

    trials = 50

    # then here we compute ransac which will give us the coefficents for our plane.
    bestPlane = f.RANSAC(points, trials)
    # we calculate the error distances between the points on the disparity and the plane.

    pointDifferences = f.calculatePointErrors(bestPlane[1], points)
    
    # print(pointDifferences)

    pointThreshold = np.average(pointDifferences)
    # compute good points.
    points = f.computePlanarThreshold(points,pointDifferences,pointThreshold)
    # print(points)
    # ● For the purposes of this assignment when a road has either curved road edges or other complexities due to the road configuration (e.g. junctions, roundabouts, road type, occlusions) report and display the road boundaries as far as possible using a polygon or an alternative pixel-wise boundary.

    # You may use any heuristics you wish to aid/filter/adjust your approach but RANSAC must be central to the detection you perform.
    
    # get the points here.
    pts = f.project3DPointsTo2DImagePoints(points);
    pts = np.array(pts, np.int32);
    pts = pts.reshape((-1,1,2));
    # pts = f.project3DPointsTo2DImagePoints(random.sample(points, 4));
    # pts = np.array(pts, np.int32);
    # pts = pts.reshape((-1,1,2));

    # When the road surface plane are detected within a stereo image it must display a red polygon on the left (colour) image highlighting where the road plane has been detected as shown in Figure 1 (see the drawing examples in the OpenCV Python Lab exercises).
    cv2.polylines(imgL,[pts],True,(0,0,255), 3);

    cv2.imshow('left image',imgL)

    # print the following.
    # filename_L.png
    # filename_R.png : road surface normal (a, b, c)

    # ● Your program must compile and work with OpenCV 3.3 on the lab PCs.
    cv2.waitKey(0);
else:
    print("-- files skipped (perhaps one is missing or not PNG)");


# close all windows

cv2.destroyAllWindows()