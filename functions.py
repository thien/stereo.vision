import cv2
import math
import numpy as np
import random
import os
import csv
from multiprocessing import Pool


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
disparity_range = cv2.imread("masks/disparity_cap.png", cv2.IMREAD_GRAYSCALE)
road_threshold_mask = cv2.imread("masks/road_threshold_mask.png", cv2.IMREAD_GRAYSCALE)
car_front_mask = cv2.imread("masks/car_front_mask.png", cv2.IMREAD_GRAYSCALE);
blackImg = cv2.imread("black.png", cv2.IMREAD_GRAYSCALE);
view_range = cv2.imread("masks/view_range.png", cv2.IMREAD_GRAYSCALE);
plane_sample = cv2.imread("masks/plane_sample.png", cv2.IMREAD_GRAYSCALE);
carmask = cv2.bitwise_and(car_front_mask,car_front_mask,mask = view_range)

# https://stackoverflow.com/questions/24814941/concave-hull-with-missing-edges
# https://github.com/pmneila/morphsnakes

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
        # img_yuv = cv2.cvtColor(images[i], cv2.COLOR_BGR2HSV)
        # # equalize the histogram of the Y channel
        # img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        # # convert the YUV image back to RGB format
        # images[i] = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        # soup up gamma

        # phi = 1
        # theta = 1
        # # Increase intensity such that
        # # dark pixels become much brighter, 
        # # bright pixels become slightly bright
        # maxIntensity = 255.0 
        # k = (maxIntensity/phi)*(images[i]/(maxIntensity/theta))**0.5
        # images[i] = np.array(k,dtype='uint8')

        
        images[i] = gammaChange(images[i], 1.4)
        # img_hsl = cv2.cvtColor(images[i], cv2.COLOR_BGR2HLS)
        hsv = cv2.cvtColor(images[i], cv2.COLOR_BGR2HSV)
        hsv[:,0,:] = 10
        images[i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


    imgL, imgR = images[0],images[1]
    return (imgL, imgR)

def greyscale(imgL,imgR):
    images = [imgL, imgR]
    for i in range(len(images)):
        images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY);
        images[i] = cv2.equalizeHist(images[i])
    imgL, imgR = images[0],images[1]
    return (imgL, imgR)

# -------------------------------------------------------------------


def calculateHistogram(img):
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    return hist

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

def capDisparity(disparity):
    cv2.bitwise_and(disparity,disparity,mask = disparity_range)
    return disparity

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

def fillAltDisparity(disparity):
    # # get non-black points in image
    # nonzeros = np.nonzero(disparity)
    # # get black points on image from getting contours.
    # ret, mask = cv2.threshold(disparity, 2, 255, cv2.THRESH_BINARY_INV)
    # blackPoints = np.nonzero(mask)
    # print(mask)

    averagePoints = []
    for i in disparity:
        mean = i[i.nonzero()].mean()
        if np.isnan(mean):
            mean = 0.0
        averagePoints.append(mean)
    
    for i in range(len(disparity)):
        for j in range(len(disparity[i])):
            if disparity[i][j] < 2:
                disparity[i][j] = averagePoints[i]
    return disparity

def baseMaskDisp(disparity):
    disparity = cv2.bitwise_and(disparity,disparity,mask = plane_sample)
    return disparity
# -------------------------------------------------------------------


def generateCenterPoint(points):
    # find middle most point in x
    # find middle most point in y
    return False

def performCanny(image):
    # image = cv2.GaussianBlur(image,(5,5),0)
    image = cv2.Canny(image,110,200)
    image = removeSmallParticles(image)
    image = cv2.bitwise_and(image,image,mask = car_front_mask)
    return image
	# image = cv2.bitwise_and(check_blurred, check_blurred, mask=mask_base)

# -------------------------------------------------------------------

def disparity3DCalculation(x,y,disparity,f,B, rgb=[]):
    if (disparity[y,x] > 0):
        # calculate corresponding 3D point [X, Y, Z]
        # stereo lecture - slide 22 + 25
        Z = (f * B) / disparity[y,x];
        X = ((x - image_centre_w) * Z) / f;
        Y = ((y - image_centre_h) * Z) / f;
        # print(x,y,z)
        # add to points

        if(len(rgb) > 0):
            return [X,Y,Z,rgb[y,x,2], rgb[y,x,1],rgb[y,x,0]]
        else:
            return [X,Y,Z]


def projectDisparityTo3d(disparity, max_disparity, rgb=[], heightRange=False):
    # list of points
    points = [];
    f = camera_focal_length_px;
    B = stereo_camera_baseline_m;
    height, width = disparity.shape[:2];
    # assume a minimal disparity of 2 pixels is possible to get Zmax
    # and then get reasonable scaling in X and Y output
    # Zmax = ((f * B) / 2);

    if heightRange != False:
        mi, ma = heightRange
        for y in range(mi, ma): # 0 - height is the y axis index
            for x in range(width): # 0 - width is the x axis index
                # if we have a valid non-zero disparity
                # points.append(disparity3DCalculation(x,y,disparity,f,B, rgb))
                if (disparity[y,x] > 0):
                    # calculate corresponding 3D point [X, Y, Z]
                    # stereo lecture - slide 22 + 25
                    Z = (f * B) / disparity[y,x];
                    X = ((x - image_centre_w) * Z) / f;
                    Y = ((y - image_centre_h) * Z) / f;
                    # print(x,y,z)
                    # add to points

                    if(len(rgb) > 0):
                        points.append([X,Y,Z,rgb[y,x,2], rgb[y,x,1],rgb[y,x,0]]);
                    else:
                        points.append([X,Y,Z]);
    else:
        for y in range(0,height-1, 2): # 0 - height is the y axis index
            for x in range(0,width-1, 2): # 0 - width is the x axis index
                # if we have a valid non-zero disparity
                # points.append(disparity3DCalculation(x,y,disparity,f,B, rgb))
                if (disparity[y,x] > 0):
                    # calculate corresponding 3D point [X, Y, Z]
                    # stereo lecture - slide 22 + 25
                    Z = (f * B) / disparity[y,x];
                    X = ((x - image_centre_w) * Z) / f;
                    Y = ((y - image_centre_h) * Z) / f;
                    # print(x,y,z)
                    # add to points

                    if(len(rgb) > 0):
                        points.append([X,Y,Z,rgb[y,x,2], rgb[y,x,1],rgb[y,x,0]]);
                    else:
                        points.append([X,Y,Z]);
    return points;

def process_list(contents):
    # print("WE ARE PROCESSING")
    if len(contents) > 3:
        disparity, max_disparity, height_range, rgb = contents[0], contents[1], contents[2], contents[3]
        results = projectDisparityTo3d(disparity, max_disparity, rgb, height_range)
    else:
        disparity, max_disparity, height_range = contents[0], contents[1], contents[2]
        results = projectDisparityTo3d(disparity, max_disparity, heightRange=height_range)
    # print("we is done")
    return results


def projectDisparityMultiProcessing(disparity, max_disparity, rgb=[]):
    # make sure we have one available.
    cpuCount = os.cpu_count()*2
    cpu_pool = Pool(cpuCount) 

    print("init split")
    # split heights so we can parallelise them.
    height, _ = disparity.shape[:2]
    inc = int(height / cpuCount)
    caps = []
    head, tail = 0, 0
    while tail < height-1:
        if tail + inc > height-1:
            tail = height - 1
        else:
            tail += inc
        height_range = (head, tail)
        if len(rgb) > 0:
            caps.append((disparity, max_disparity, height_range, rgb))
        else:
            caps.append((disparity, max_disparity, height_range))
        head = tail+1

    print("doing multiprocessing..")
    pool_outputs = cpu_pool.map(process_list, caps)
    print("closing taks")
    cpu_pool.close() # no more tasks
    print("wrap tasks")
    cpu_pool.join()  # wrap up current tasks
    print("done multiprocessing")
    return pool_outputs


def printNormalString(normal):
    print("Normal:", str(round(normal[0][0],4))+"x +", str(round(normal[1][0],4))+"y +", str(round(normal[2],4)) + "z")

def drawContours(image, points):
    # generate convex hull
    hull = cv2.convexHull(points)
    # draw hull on image
    return (cv2.drawContours(image,[hull],0,(0,0,255),5), hull)

def getNormalVectorLine(basePoint, abc, disparity):
    f = camera_focal_length_px;
    B = stereo_camera_baseline_m;
    # basepoint is x,y
    x,y = basePoint

    Z = (f * B) / disparity[y,x];
    X = ((x - image_centre_w) * Z) / f;
    Y = ((y - image_centre_h) * Z) / f;

    newY = Y - 0.505
    newX = X + 0.205
    newZ = False

    d = math.sqrt(abc[0]*abc[0]+abc[1]*abc[1]+abc[2]*abc[2])
    Z = d - ((abc[0] * newX) + (abc[1]*newY))

    newX = ((newX * camera_focal_length_px) / Z) + image_centre_w;
    newY = ((newY * camera_focal_length_px) / Z) + image_centre_h;

    results = (int(newX), int(newY[0]))
    return results


def RGBToGreyscale(r,g,b):
    # we use some modified YLinear weights sourced from (https://en.wikipedia.org/wiki/Grayscale)
    return int(0.06*r + 0.75*g + 0.19*b)

def getPointColour(point):
    return (point[3], point[4], point[5])

def calculateColourHistogram(points):
    # get colour points for each point in plane.
    # convert it to greyscale
    # colours = [RGBToGreyscale(pt[3],pt[4],pt[5]) for pt in points]
    colours = [tuple([pt[3],pt[4],pt[5]]) for pt in points]

    histogram = {}
    for i in colours:
        if i not in histogram:
            histogram[i] = 1
        else:
            histogram[i] += 1
    return histogram

def filterPointsByHistogram(points, histogram, threshold=100):
    # sort keys by histogram rank
    # sortedColours = sorted(histogram, key=histogram.__getitem__)
    # for colour in sortedColours:
    #     print( colour, 'corresponds to', histogram[colour])

    # pythonic expression for filtering 
    points = [x for x in points if histogram[getPointColour(x)] > threshold]

    return points


# -------------------------------------------------------------------

# project a set of 3D points back the 2D image domain
def project3DPointsTo2DImagePoints(points):
    points2 = [];
    # calc. Zmax as per above
    # Z = (camera_focal_length_px * stereo_camera_baseline_m) / 2;
    for i1 in range(len(points)):
        # reverse earlier projection for X and Y to get x and y again
        Z = points[i1][2]
        x = ((points[i1][0] * camera_focal_length_px) / Z) + image_centre_w;
        y = ((points[i1][1] * camera_focal_length_px) / Z) + image_centre_h;
        points2.append([x,y]);
    return points2;

# -------------------------------------------------------------------

def printVectorPlane(plane=[]):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    point  = np.array([1, 2, 3])
    normal = np.array([1, 1, 2])

    # a plane is a*x+b*y+c*z+d=0
    # [a,b,c] is the normal. Thus, we have to calculate
    # d and we're set
    d = -point.dot(normal)

    # create x,y
    xx, yy = np.meshgrid(range(10), range(10))

    # calculate corresponding z
    z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]

    # plot the surface
    plt3d = plt.figure().gca(projection='3d')
    plt3d.plot_surface(xx, yy, z)
    plt.show()


# ● For the purposes of this assignment when a road has either curved road edges or other complexities due to the road configuration (e.g. junctions, roundabouts, road type, occlusions) report and display the road boundaries as far as possible using a polygon or an alternative pixel-wise boundary.

# You may use any heuristics you wish to aid/filter/adjust your approach but RANSAC must be central to the detection you perform.

# http://pathfinder.engin.umich.edu/documents/Feng&Taguchi&Kamat.ICRA.2014.pdf
# http://web.ipac.caltech.edu/staff/fmasci/home/astro_refs/HoughTrans_lines_09.pdf
# https://stackoverflow.com/questions/18255958/harris-corner-detection-and-localization-in-opencv-with-python
# https://stackoverflow.com/questions/32609098/how-to-fast-change-image-brightness-with-python-opencv


def randomNonCollinearPoints(points):
    # print("Calculating Non CollinearPoints")
    # ---------------------------------------------------------------
    # how to - select 3 non-colinear points
    cross_product_check = np.array([0,0,0]);

    # cross product checks
    cp_check0 = True if cross_product_check[0] == 0 else False
    cp_check1 = True if cross_product_check[1] == 0 else False
    cp_check2 = True if cross_product_check[2] == 0 else False
    
    P1 = None
    P2 = None
    P3 = None

    while cp_check0 and cp_check1 and cp_check2:
        P1 = np.array([x[:3] for x in random.sample(points, 1)])[0]
        P2 = np.array([x[:3] for x in random.sample(points, 1)])[0]
        P3 = np.array([x[:3] for x in random.sample(points, 1)])[0]
        # make sure they are non-collinear
        # print("Checking for colineararity..")
        cross_product_check = np.cross(P1-P2, P2-P3)

        cp_check0 = True if cross_product_check[0] == 0 else False
        cp_check1 = True if cross_product_check[1] == 0 else False
        cp_check2 = True if cross_product_check[2] == 0 else False
    # print("Calculated Non-Colinear Points.")
    return (P1, P2, P3)


def getBlackImage():
    return blackImg


def planarFitting(randomPoints, points):
    # [Further hint: for this assignment this can be done in full projected floating-point 3D space (X,Y,Z) or in integer image space (x,y,disparity) – see provided hints python file]
    
    # generate random colinear points.
    # Here we choose from the whole point range
    P1, P2, P3 = randomNonCollinearPoints(points)

    # calculate coefficents a,b, and c
    abc = np.dot(np.linalg.inv(np.array([P1,P2,P3])), np.ones([3,1]))

    # calculate coefficents d
    d = math.sqrt(abc[0]*abc[0]+abc[1]*abc[1]+abc[2]*abc[2])


    if len(randomPoints[0]) > 3:
        randomPoints = [[item[0], item[1], item[2]] for item in randomPoints]

    # measure distance of our random points from plane given 
    # the plane coefficients calculated
    
    dist = abs((np.dot(randomPoints, abc) - 1)/d)

    # calculate the normal
    normal = (abc[0],abc[1],-1)
    nn = np.linalg.norm(normal)
    normal = normal / nn

    return abc, normal, dist


def getCenterPoint(points):
    (x,y),_ = cv2.minEnclosingCircle(points)
    return (int(x),int(y))

# removes artifacts.
def removeSmallParticles(image, threshold=20):
    _, contours, _= cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for spectacle in contours:
        area = cv2.contourArea(spectacle)
        if area < threshold:
            # its most likely a particle, colour it black.
            cv2.drawContours(image,[spectacle],0,0,-1)
    return image

def generatePointsAsImage(points):
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    img = blackImg.copy()
    # draw points on image.
    for i in points:
        img[i[0][1]][i[0][0]] = 255

    return img

def sanitiseRoadImage(img, size):
    (height, width) = size
    referenceImg = img.copy()
    # perform closing on the image to fill holes
    kernel = np.ones((9,9),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # erode a little bit.
    img = cv2.erode(img,kernel,iterations = 2)

    # put a threshold for the road points (used for convex hull purposes)
    img = cv2.bitwise_and(img,img,mask = road_threshold_mask)
    # # Generate est. Border.
    # bigKernel = np.ones((15,15),np.uint8)
    # Borders = cv2.dilate(img,bigKernel,iterations = 1)
    # Borders = cv2.erode(Borders,bigKernel,iterations = 1)

    # Borders = removeSmallParticles(Borders, 200)

    # blurredImg = cv2.GaussianBlur(img,(15,15),0)
    # img = cv2.bitwise_and(img, img, mask=referenceImg)
    pts = []
    for i in range(height):
        for j in range(width):
            if img[i][j] != 0:

                k = [j,i]
                # print(k)
                pts.append(k)
    pts = np.matrix(pts)
    return img, pts
# -------------------------------------------------------------------

def RANSAC(points, trials):
    # Your solution must use a RANdom SAmple and Consensus (RANSAC) approach to perform the detection of the 3D plane in front of the vehicle (when and where possible).
    bestPlane = (None, None)
    bestError = float("inf")
    
    for i in range(trials):
        # select T data points randomly
        T = random.sample(points, 600)
        # estimate the plane using this subset of information
        coefficents, normal, dist = planarFitting(T, points)
        error = np.mean(dist)
        
        # store the results in our dictionary.
        if error < bestError:
            bestPlane = (normal,coefficents)
            bestError = error
            # print("New Best Error:", error)
    return bestPlane

# -------------------------------------------------------------------

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
        # else:
        #     print("Doesn't meet threshold:", differences[i])
    return new_points

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

# -------------------------------------------------------------------

def generatePlaneShape(points, copy):
    img = cv2.cvtColor(copy,cv2.COLOR_BGR2GRAY)
    img[:] = 0
    for i in points:
        # print(i)
        img[i[0][1]][i[0][0]] = 255
    kernel = np.ones((3,3),np.uint8)
    img = cv2.erode(img,kernel,iterations = 1)
    img = ParticleCleansing(img)
    return img

def ParticleCleansing(image):
	"""
	ParticleCleansing
	Searches for particles in an images and removes it using contours.
	@param image: thresholded image
	@param image: sanitised image
	"""
	_, contours, _= cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	for spot in contours:
		area = cv2.contourArea(spot)
		if area < 20:
			cv2.drawContours(image,[spot],0,0,-1)
	return image

# -------------------------------------------------------------------

def getPtsAgain(plane_shape):
    pts = []
    for i in range(len(plane_shape)):
        for j in range(len(plane_shape[i])):
            if plane_shape[i][j] != 0:
                pts.append([[i,j]])
    pts = np.array(pts).astype("uint8")
    return pts  

def drawConvexHull(pts, base, thickness=1, colour=(0,0,255)):
    """
    Draws the convex hull of an image coordinates to a base image.
    """

    # https://docs.opencv.org/3.3.1/dd/d49/tutorial_py_contour_features.html
    # epsilon = 1*cv2.arcLength(pts,True)
    # approx = cv2.approxPolyDP(pts,epsilon,True)
    hull = cv2.convexHull(pts)
    # draw the convex hull 
    cv2.drawContours(base,[hull],0,colour,thickness)
    return base

# -------------------------------------------------------------------

# plotting of the planar normal direction direction glyph / vector in the image
def drawNormalGlyph(normal):
    return False

# automatically adjusting the parameters initial region of interest extraction or image prefiltering based on some form of preliminary analysis image
def filterRegionByPopularColour(image):
    return False

# the successful use of any heuristics to speed up processing times or reduce false detections (including at the stage of colour pre-filtering)

# automatically detecting and highlighting obstacles that rise above the road surface plane (vehicles, pedestrians, bollards etc.) as they appear directly in front of the vehicles
def detectObjects(image):
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)

    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)

    # Threshold for an optimal value, it may vary depending on the image.
    image[dst>0.01*dst.max()]=[255,0,0]

    return image

# -------------------------------------------------------------------



def drawNormalLine(baseImage, center, normal, disparity):
    newLine = getNormalVectorLine(center, normal, disparity)
    lineThickness = 2
    normalLineColor = (20,185,255)
    cv2.line(baseImage, center, newLine, normalLineColor, lineThickness)
    circleHeadColour = (normalLineColor[0]+10, normalLineColor[1]+10, normalLineColor[2]+10)
    cv2.circle(baseImage, newLine, 2, circleHeadColour, thickness=10, lineType=8, shift=0)
    return baseImage

def resizeImage(image, height, width):
    image = cv2.resize(image, (width, height))
    return image

def batchImages(imgList, size):
    # for each image, resize them.
    counts = len(imgList)

    # resize all images s.t they all fit accordingly.
    images = []
    for i in imgList:
        fixedScaleImg = resizeImage(i[1], size[0], size[1])
        # check if greyscale
        dim = len(fixedScaleImg.shape)
        if dim != 3:
            # convert to colour.
            fixedScaleImg = cv2.cvtColor(fixedScaleImg, cv2.COLOR_GRAY2BGR)
        # add to list.
        images.append(fixedScaleImg)
        


    # numpy_vertical = np.vstack((image, grey_3_channel))
    # numpy_horizontal = np.hstack((image, grey_3_channel))

    # numpy_vertical_concat = np.concatenate((image, grey_3_channel), axis=0)
    # numpy_horizontal_concat = np.concatenate((image, grey_3_channel), axis=1)
    if counts == 1:
        return images[0]
    if counts == 2:
        stack = np.hstack((images[0], images[1]))
        return cv2.resize(stack, (0,0), fx=0.5, fy=0.5) 
    if counts == 3:
        p1 = np.hstack((images[0], images[1]))
        p2 = np.hstack((images[2], images[2]))
        stack = np.vstack((p1, p2))
        return cv2.resize(stack, (0,0), fx=0.5, fy=0.5) 
    if counts == 4:
        p1 = np.hstack((images[0], images[1]))
        p2 = np.hstack((images[2], images[3]))
        stack = np.vstack((p1, p2))
        return cv2.resize(stack, (0,0), fx=0.5, fy=0.5)
    if counts == 4:
        p1 = np.hstack((images[0], images[1]))
        p2 = np.hstack((images[2], images[3]))
        stack = np.vstack((p1, p2))
        return cv2.resize(stack, (0,0), fx=0.5, fy=0.5)
    if counts == 5:
        p1 = np.hstack((images[0], images[1]))
        p2 = np.hstack((images[2], images[3]))
        stack = np.vstack((p1, p2))
        l = cv2.resize(stack, (0,0), fx=0.25, fy=0.25)
        r =  cv2.resize(images[4], (0,0), fx=0.5, fy=0.5)
        r = np.hstack((l, r))
        return r
    else:
        pass



# def batchViewImages(images):

#     return False

# def batchViewImages(images):
#     f, axarr = plt.subplots(2,2)
#     axarr[0,1].imshow(imgR2)
#     axarr[0,0].imshow(imgL2)
#     axarr[1,0].imshow(disparity_scaled2)
#     plt.show()

# def batchViewImages(images):
#     # translated from C code here:
#     # https://github.com/opencv/opencv/wiki/DisplayManyImages
#     size = 0
#     i = 0
#     x = 0
#     y = 0
#     if len(images) == 1:
#         w = h = 1
#         size = 300
#     elif len(images) == 2:
#         w = 2
#         h = 1
#         size = 300
#     elif len(images) == 3 or len(images) == 4:
#         w = h = 2
#         size = 300
#     elif len(images) == 5 or len(images) == 6:
#         w = 3
#         h = 2
#         size = 200
#     else:
#         w = 4
#         h = 3
#         size = 150
    
#     dimension = (w, h, channels) = ((100 + size * w), (60+size*h), 3)
#     img = np.zeros(dimension, np.int8)

#     m = 0
#     n = 0
#     for i in range(len(images)):
