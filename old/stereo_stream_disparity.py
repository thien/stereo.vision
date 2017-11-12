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
import extra as ex
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

# display full or cropped disparity image
crop_disparity = False;
# pause until key press after each image
pause_playback = False;


#####################################################################

# resolve full directory location of data set for left / right images

path_dir_l =  os.path.join(dataset_path, directory_to_cycle_left);
path_dir_r =  os.path.join(dataset_path, directory_to_cycle_right);

# get a list of the left image files and sort them (by timestamp in filename)

filelist_l = sorted(os.listdir(path_dir_l));

# setup the disparity stereo processor to find a maximum of 128 disparity values
# (adjust parameters if needed - this will effect speed to processing)
max_disparity = 32;
stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21);

# carmask = cv2.imread("mask.png", cv2.IMREAD_COLOR)
carmask = cv2.imread("masks/car_front_mask.png", cv2.IMREAD_GRAYSCALE);
view_range = cv2.imread("masks/view_range.png", cv2.IMREAD_GRAYSCALE);
carmask = cv2.bitwise_and(carmask,carmask,mask = view_range)

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

        # from the left image filename get the correspondoning right image

        filename_right = filename_l.replace("_L", "_R");
        full_path_filename_l = os.path.join(path_dir_l, filename_l);
        full_path_filename_r = os.path.join(path_dir_r, filename_right);

        # check the file is a PNG file (left) and check a correspondoning right image actually exists

        if ('.png' in filename_l) and (os.path.isfile(full_path_filename_r)) :

            # read left and right images and display in windows
            # N.B. despite one being grayscale both are in fact stored as 3-channel
            # RGB images so load both as such

            imgL = cv2.imread(full_path_filename_l, cv2.IMREAD_COLOR)
            imgR = cv2.imread(full_path_filename_r, cv2.IMREAD_COLOR)

            # for sanity print out these filenames
            print(full_path_filename_l);
            print(full_path_filename_r);
            print();

            print("-- files loaded successfully");
            print();

            # remember to convert to grayscale (as the disparity matching works on grayscale)
            # N.B. need to do for both as both are 3-channel images

            grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY);
            grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY);
            #  grayL = cv2.bitwise_and(grayL,grayL,mask = carmask)
            # grayR = cv2.bitwise_and(grayR,grayR,mask = carmask)
            # compute disparity image from undistorted and rectified stereo images that we have loaded
            # (which for reasons best known to the OpenCV developers is returned scaled by 16)

            disparity = stereoProcessor.compute(grayL,grayR);
 
            disparity = cv2.bitwise_and(disparity,disparity,mask = carmask)

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

            # crop disparity to chop out left part where there are with no disparity
            # as this area is not seen by both cameras and also
            # chop out the bottom area (where we see the front of car bonnet)

            if (crop_disparity):
                width = np.size(disparity_scaled, 1);
                disparity_scaled = disparity_scaled[0:390,135:width];

            # display image (scaling it to the full 0->255 range based on the number
            # of disparities in use for the stereo part)

            # fill the empty parts in disparity
            if previousDisparity is not None:
            #     for i in range(0,len(disparity_scaled)):
            #         for j in range(0,len(disparity_scaled[i])):
            #             if disparity_scaled[i][j] == 0:
            #                 disparity_scaled[i][j] = previousDisparity[i][j]
                ret, mask = cv2.threshold(disparity_scaled, 2, 255, cv2.THRESH_BINARY)
                mask = cv2.bitwise_not(mask)

                # Take only region of logo from logo image.
                filling = cv2.bitwise_and(previousDisparity,previousDisparity,mask = mask)
                disparity_scaled = cv2.add(disparity_scaled,filling)
                cv2.imshow("original Image",imgL)
            cv2.imshow("disparity", (disparity_scaled * (256. / max_disparity)).astype(np.uint8));


            ex.handleKey(cv2, pause_playback, disparity_scaled, imgL, imgR, crop_disparity)
            # store the disparity
            previousDisparity = disparity_scaled
        else:
            print("-- files skipped (perhaps one is missing or not PNG)");

except Exception as error:
    print("Exception:", error)
# close all windows

cv2.destroyAllWindows()