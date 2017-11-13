"""
Task Specification – Road Surface Region Detection

    You are required develop to road surface detection system that correctly detects the 3D planar orientation and bounds (i.e. edges) of any free (unoccupied) road surface region immediately in- front of the vehicle in which the autonomous vehicle needs to operate (e.g. for staying in the correct lane / staying on the road itself / avoiding obstacles / automatic braking).

    In constructing your solution you may wish to consider the following aspects of your design:
    • exploring the optimization of the provided stereo vision algorithm in use and its operation and hows its performance under varying illumination conditions could perhaps be improved using the HSV or other colour spaces covered in CS SM L2 - Image Processing (these paper also presents an interesting research approach [1,2], others solutions exist also for illumination invariant colour spaces; use these as a starting point for your search).

    • selection of a region of interest, possibly adaptively, within the image (including possibly areas of road, pavement, other or not) that represents the region directly in-front of the vehicle and how to deal with the problem of missing disparity (depth) values in that region.

    • Calculating the equation of a plane from 3 points in 3D: http://mathworld.wolfram.com/Plane.html

    [Further hint: for this assignment this can be done in full projected floating-point 3D space (X,Y, Z)or in integer image space (x,y,disparity) – see provided hints python file]

    Your solution must use a RANdom SAmple and Consensus (RANSAC) approach to perform the detection of the 3D plane in front of the vehicle (when and where possible). For the avoidance of doubt, no credit will be given for a 2D solution based on the built-in Hough Transform or Douglas- Pecker contour detection in OpenCV or that does not recover the 3D parameters of the plane.

    Additionally, some example images may not have significant noise-free disparity (depth) available in front of the vehicle or the road region may be partially occluded by other objects (people, vehicles etc.). The road surface itself will change in terrain type, illumination conditions and road markings – ideally your solution should be able to cope with all of these. Road edges may or may not be marked by line markings in the colour image. All examples will contain a clear front facing view of the road in front of the vehicle only – your system should report all appropriate road surface plane instances it can detect recognising this may not be possible for all cases within the data set provided.
"""
"""
    http://resources.mpi-inf.mpg.de/TemporalStereo/articleJpeg.pdf
    Method:
    - optimise disparity
    - do a mask on the disparity
    - do stereo to 3D
    - delete coordinates of points that are blank
    - planar fitting ransac
    - create a plane accordingly
    - create polygon of points near the plane
    - map polygon of points on the plane
        - convex hull
    - show coefficents on the image
"""
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
max_disparity = 32;

# Start the loop
try:
    previousDisparity = None
    for filename_l in filelist_l:
        """
        Here we'll cycle through the files, and finding each stereo pair.
        We'll then process them to detect the road surface planes, and compute 
        the stereo disparity.
        """

        # skip forward to start a file we specify by timestamp (if this is set)
        if ((len(skip_forward_file_pattern) > 0) and not(skip_forward_file_pattern in filename_l)):
            continue;
        elif ((len(skip_forward_file_pattern) > 0) and (skip_forward_file_pattern in filename_l)):
            skip_forward_file_pattern = "";

        # # from the left image filename get the corresponding right image
        imageStores = f.loadImages(filename_l, path_dir_l, path_dir_r)
        if imageStores != False:
            imgL, imgR = imageStores

            imgL, imgR = f.preProcessImages(imgL,imgR)
            grayL, grayR = f.greyscale(imgL,imgR)

            skeleton = f.performCanny(imgL)
            cv2.imshow('canny', skeleton)

            # compute disparity image from undistorted and rectified stereo images that we have loaded
            disparity = f.disparity(grayL, grayR, max_disparity, crop_disparity)
            disparity = f.maskDisparity(disparity)

            # load previous disparity to fill in missing content.
            disparity = f.fillDisparity(disparity, previousDisparity)
            previousDisparity = disparity

            # show disparity
            cv2.imshow("disparity", disparity);



            # project to a 3D colour point cloud (with or without colour)
            points = f.projectDisparityTo3d(disparity, max_disparity, imgL);

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







            cv2.polylines(imgL,[pts],True,(0,0,255), 3);


            # convert disparity to rgb so we can map it with the image.
            # backtorgb = cv2.cvtColor(disparity,cv2.COLOR_GRAY2RGB)

            # imgL = cv2.bitwise_xor(imgL, backtorgb)
            
            cv2.imshow('left image',imgL)



            # ● Your program must compile and work with OpenCV 3.3 on the lab PCs.
            f.handleKey(cv2, pause_playback, disparity, imgL, imgR, crop_disparity)
        else:
            print("-- files skipped (perhaps one is missing or not PNG)");

except Exception as error:
    print("Exception:", error)
# close all windows

cv2.destroyAllWindows()