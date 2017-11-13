import cv2
import math
import numpy as np
import random
import os
import csv

# focal length in pixels
camera_focal_length_px = 399.9745178222656
# focal length in metres (4.8 mm) 
camera_focal_length_m = 4.8 / 1000
# camera baseline in metres    
stereo_camera_baseline_m = 0.2090607502

image_centre_h = 262.0;
image_centre_w = 474.5;

max_disparity = 128;
stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21);
    
# Here we load pre-requisite masks. This way, they're preloaded
# and ready for when they are needed on an image.
car_front_mask = cv2.imread("masks/car_front_mask.png", cv2.IMREAD_GRAYSCALE);
view_range = cv2.imread("masks/view_range.png", cv2.IMREAD_GRAYSCALE);
carmask = cv2.bitwise_and(car_front_mask,car_front_mask,mask = view_range)

# -------------------------------------------------------------------

def loadImages(filename_l, path_dir_l, path_dir_r):
    """
    Here we'll cycle through the files, and finding each stereo pair.
    We'll then process them to detect the road surface planes, and compute 
    the stereo disparity.
    """
    # from the left image filename get the correspondoning right image
    filename_right = filename_l.replace("_L", "_R");
    full_path_filename_l = os.path.join(path_dir_l, filename_l);
    full_path_filename_r = os.path.join(path_dir_r, filename_right);
    # check the file is a PNG file (left) and check a corresponding right image actually exists
    if ('.png' in filename_l) and (os.path.isfile(full_path_filename_r)):
        # read left and right images and display in windows
        # N.B. despite one being grayscale both are in fact stored as 3-channel
        # RGB images so load both as such
        imgL = cv2.imread(full_path_filename_l)
        imgR = cv2.imread(full_path_filename_r)

        # for sanity print out these filenames
        print(full_path_filename_l);
        print(full_path_filename_r);
        print();

        return (imgL, imgR)
    else:
        return False

# -------------------------------------------------------------------

def gammaChange(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

# -------------------------------------------------------------------

def preProcessImages(imgL,imgR):

    images = [imgL, imgR]
    for i in range(len(images)):
        # https://www.packtpub.com/packtlib/book/Application-Development/9781785283932/2/ch02lvl1sec26/Enhancing%20the%20contrast%20in%20an%20image
        img_yuv = cv2.cvtColor(images[i], cv2.COLOR_BGR2YUV)
        # equalize the histogram of the Y channel
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        # convert the YUV image back to RGB format
        images[i] = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        # soup up gamma

        phi = 1
        theta = 1
        # Increase intensity such that
        # dark pixels become much brighter, 
        # bright pixels become slightly bright
        maxIntensity = 255.0 
        k = (maxIntensity/phi)*(imgL/(maxIntensity/theta))**0.5
        imgL = np.array(k,dtype='uint8')


        images[i] = gammaChange(images[i], 1.4)
        


    imgL, imgR = images[0],images[1]
    return (imgL, imgR)

def greyscale(imgL,imgR):
    images = [imgL, imgR]
    for i in range(len(images)):
        images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY);
    imgL, imgR = images[0],images[1]
    return (imgL, imgR)

# -------------------------------------------------------------------

# compute disparity image from undistorted and rectified stereo images that we have loaded
# (which for reasons best known to the OpenCV developers is returned scaled by 16)
def disparity(grayL, grayR, max_disparity, crop_disparity):
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

    # crop disparity to chop out left part where there are with no disparity
    # as this area is not seen by both cameras and also
    # chop out the bottom area (where we see the front of car bonnet)

    if (crop_disparity):
        width = np.size(disparity_scaled, 1);
        disparity_scaled = disparity_scaled[0:390,135:width];

    # display image (scaling it to the full 0->255 range based on the number
    # of disparities in use for the stereo part)

    disparity_scaled = (disparity_scaled * (256. / max_disparity)).astype(np.uint8)
    return disparity_scaled

# -------------------------------------------------------------------

def maskDisparity(disparity):
    #     # Take only region
    #     filling = cv2.bitwise_and(previousDisparity,previousDisparity,mask = carmask)
    #     disparity = cv2.add(disparity,filling)
    disparity = cv2.bitwise_and(disparity,disparity,mask = carmask)
    return disparity

def fillDisparity(disparity, previousDisparity):
    if previousDisparity is not None:
        ret, mask = cv2.threshold(disparity, 2, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_not(mask)
        # Take only region of logo from logo image.
        filling = cv2.bitwise_and(previousDisparity,previousDisparity,mask = mask)
        disparity = cv2.add(disparity,filling)
    return disparity

# -------------------------------------------------------------------

# removes artifacts.
def removeSmallParticles(image):
    _, contours, _= cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for spectacle in contours:
        area = cv2.contourArea(spectacle)
        if area < 20:
            # its most likely a particle, colour it black.
            cv2.drawContours(image,[spectacle],0,0,-1)
    return image

def performCanny(image):
    # image = cv2.GaussianBlur(image,(5,5),0)
    image = cv2.Canny(image,110,200)
    image = removeSmallParticles(image)
    image = cv2.bitwise_and(image,image,mask = car_front_mask)
    return image
	# image = cv2.bitwise_and(check_blurred, check_blurred, mask=mask_base)

# -------------------------------------------------------------------

# project_disparity_to_3d : project a given disparity image
# (uncropped, unscaled) to a set of 3D points with optional colour
def projectDisparityTo3d(disparity, max_disparity, rgb=[]):
    # print("projecting disparity")
    # list of points
    points = [];
    f = camera_focal_length_px;
    B = stereo_camera_baseline_m;
    height, width = disparity.shape[:2];
    # assume a minimal disparity of 2 pixels is possible to get Zmax
    # and then get reasonable scaling in X and Y output
    Zmax = ((f * B) / 2);
    for y in range(height): # 0 - height is the y axis index
        for x in range(width): # 0 - width is the x axis index
            # if we have a valid non-zero disparity
            if (disparity[y,x] > 0):
                # calculate corresponding 3D point [X, Y, Z]
                # stereo lecture - slide 22 + 25
                Z = (f * B) / disparity[y,x];
                X = ((x - image_centre_w) * Zmax) / f;
                Y = ((y - image_centre_h) * Zmax) / f;
                # print(x,y,z)
                # add to points
                if(rgb.size > 0):
                    points.append([X,Y,Z,rgb[y,x,2], rgb[y,x,1],rgb[y,x,0]]);
                else:
                    points.append([X,Y,Z]);
    return points;

# -------------------------------------------------------------------

# project a set of 3D points back the 2D image domain
def project3DPointsTo2DImagePoints(points):
    points2 = [];
    # calc. Zmax as per above
    Zmax = (camera_focal_length_px * stereo_camera_baseline_m) / 2;
    for i1 in range(len(points)):
        # reverse earlier projection for X and Y to get x and y again
        x = ((points[i1][0] * camera_focal_length_px) / Zmax) + image_centre_w;
        y = ((points[i1][1] * camera_focal_length_px) / Zmax) + image_centre_h;
        points2.append([x,y]);
    return points2;

# -------------------------------------------------------------------

# select a random subset of the 3D points (4 in total)
# and them project back to the 2D image (as an example)
# ● For the purposes of this assignment when a road has either curved road edges or other complexities due to the road configuration (e.g. junctions, roundabouts, road type, occlusions) report and display the road boundaries as far as possible using a polygon or an alternative pixel-wise boundary.

# You may use any heuristics you wish to aid/filter/adjust your approach but RANSAC must be central to the detection you perform.

def soupedUpPlaneFitting(points):
    # perform fitting algorithm
    matA = []
    matB = []

    for i in range(len(points)):
        matA.append([points[i][0], points[i][1], 1])
        matB.append(points[i][2])

    A = np.matrix(matA)
    # transpose B so we can matrix operation for quicktimes
    b = np.matrix(matB).T 
    
    abc = (A.T * A).I * A.T * b
    # calculate errors for each point
    errors = b - A * abc
    # at this stage we should throw away coordinates that
    # have a large enough error rate.

    # print ("%f x + %f y + %f = z" % (abc[0], abc[1], abc[2]))
    
    # calculating the normal of this plane.
    normal = (abc[0],abc[1],-1)
    nn = np.linalg.norm(normal)
    normal = normal / nn
    # print("Normal:", normal)
    # calculate average error value here.
    # errorValue = np.average(errors)
    # calculate sum of errors
    errorSum = np.sum(errors)
    # print ("Plane Error Average:",errorValue)
    return (errorSum, normal, abc)
 
def calculatePointErrors(abc, points):
    the_list = []
    for i in points:
        p = [i[0],i[1],i[2]]
        # print(p)
        the_list.append(p)
    points = np.array(the_list)

    # calculate coefficents d
    d = math.sqrt(abc[0]*abc[0]+abc[1]*abc[1]+abc[2]*abc[2])

    # measure distance of all points from plane given 
    # the plane coefficients calculated
    dist = abs((np.dot(points, abc) - 1)/d)
    return dist

def computePlanarThreshold(points,differences,threshold=0.01):
    """
        Discards points on the disparity where it is not within the plane.
    """
    new_points = []
    for i in range(len(points)):
        # we only keep points that are within the threshold.
        if differences[i] < threshold:
            new_points.append(points[i])
    return new_points

def computePlanarFitting(points):
    # points = np.array(....) ... of 3D points

    # https://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points
    # Calculating the equation of a plane from 3 points in 3D: http://mathworld.wolfram.com/Plane.html


    # [Further hint: for this assignment this can be done in full projected floating-point 3D space (X,Y,Z) or in integer image space (x,y,disparity) – see provided hints python file]
    # ....

    # ---------------------------------------------------------------
    # how to - select 3 non-colinear points
    cross_product_check = np.array([0,0,0]);

    # cross product checks
    cp_check0 = True if cross_product_check[0] == 0 else False
    cp_check1 = True if cross_product_check[1] == 0 else False
    cp_check2 = True if cross_product_check[2] == 0 else False
    
    while cp_check0 and cp_check1 and cp_check2:
        [P1,P2,P3] = points[random.sample(xrange(len(points)), 3)]
        # make sure they are non-collinear
        cross_product_check = np.cross(P1-P2, P2-P3)

    # ---------------------------------------------------------------
    # how to - calculate plane coefficients from these points

    # calculate coefficents a,b, and c
    abc = np.dot(np.linalg.inv(np.array([P1,P2,P3])), np.ones([3,1]))

    # calculate coefficents d
    d = math.sqrt(abc[0]*abc[0]+abc[1]*abc[1]+abc[2]*abc[2])

    # how to - 
    # measure distance of all points from plane given 
    # the plane coefficients calculated
    dist = abs((np.dot(points, abc) - 1)/d)

    return dist


# -------------------------------------------------------------------

# Your solution must use a RANdom SAmple and Consensus (RANSAC) approach to perform the detection of the 3D plane in front of the vehicle (when and where possible).

def RANSAC(points, trials):
    """
        In computer vision a standard way is to use RANSAC or MSAC, in your case;
        Take 3 random points from the population
        Calculate the plane defined by the 3 points
        Sum the errors (distance to plane) for all of the points to that plane.
        Keep the 3 points that show the smallest sum of errors (and fall within a threshold).
        Repeat N iterations (see RANSAC theory to choose N, may I suggest 50?)
    """

    planes = {}
    for i in range(1,trials):
        # select T data points randomly
        T = random.sample(points, 8)
        # print("we have points",T)
        # estimate feature parameters using this model
        # (errors, normal, coefficents)
        error, normal, abc = soupedUpPlaneFitting(T)
        # errorRate = computePlanarFitting(points)
        planes[error] = (normal, abc)
    
    # get the plane with the smallest errors
    keylist = sorted(planes)
    # planes = sorted(planes.iterkeys())
    the_plane = planes[keylist[0]]
    print(the_plane)
    return the_plane

# -------------------------------------------------------------------

# write to file in an X simple ASCII X Y Z format that can be viewed in 3D
# using the on-line viewer at http://lidarview.com/
# (by uploading, selecting X Y Z format, press render , rotating the view)
def saveCoords(points, file_location):
    # point_cloud_file = open('3d_points.txt', 'w');
    point_cloud_file = open(file_location, 'w');
    csv_writer = csv.writer(point_cloud_file, delimiter=' ');
    csv_writer.writerows(points);
    point_cloud_file.close();

# -------------------------------------------------------------------

def handleKey(cv2, pause_playback, disparity_scaled, imgL, imgR, crop_disparity):
    # keyboard input for exit (as standard), save disparity and cropping
        # exit - x
        # save - s
        # crop - c
        # pause - space

    # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)
    key = cv2.waitKey(2 * (not(pause_playback))) & 0xFF;
    if (key == ord('s')):     # save
        cv2.imwrite("sgbm-disparty.png", disparity_scaled);
        cv2.imwrite("left.png", imgL);
        cv2.imwrite("right.png", imgR);
    elif (key == ord('c')):     # crop
        crop_disparity = not(crop_disparity);
    elif (key == ord(' ')):     # pause (on next frame)
        pause_playback = not(pause_playback);
    elif (key == ord('x')):       # exit
        raise ValueError("exiting manually")