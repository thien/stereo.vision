# imports, don't touch them lol
import cv2
import numpy as np
# import csv
import functions as f
import datetime
import sys
import traceback

# (adjust parameters if needed - this will effect speed to processing)
default_opts = {
    'crop_disparity' : False,       # display full or cropped disparity image
    'pause_playback' : False,       # pause until key press after each image
    'max_disparity' : 128,
    'ransac_trials' : 600,
    'road_color_thresh': 20,        # remove points from roadpts if it isn't in the x most populous colours 
    'loop': True,
    'point_threshold' : 0.03,
    'img_size' : (544,1024),
    'threshold_option' : 'previous', # options are: 'previous' or 'mean'
    'record_video' : False,
    'video_filename' : 'previous.avi'
}

def performStereoVision(imgL,imgR, prev_disp=None, opt=default_opts):
    # initiate images list.
    images = []

    # ------------------------------
    # 1. IMAGE PROCESSING
    # ------------------------------

    # perform preprocessing on images.
    imgL, imgR = f.preProcessImages(imgL,imgR)
    # make image greyscale
    grayL, grayR = f.greyscale(imgL,imgR)

    # ------------------------------
    # 2. DISPARITY PROCESSING
    # ------------------------------

    # compute disparity image
    disparity = None
    try:
        # generate disparity
        disparity = f.disparity(grayL,grayR, opt['max_disparity'], opt['crop_disparity'])
        # clean holes in the disparity
        disparity = f.disparityCleaning(disparity, opt['threshold_option'], prev_disp)
        # save the disparity and return that for the next iteration in the loop.
        prev_disp = disparity
    except Exception as e:
        # if theres an error in cleaning the disparity, return a black image.
        print("Cannot compute the disparity.")
        disparity = f.getBlackImage()

    # add disparity to list of images.
    images.append(("Disparity",disparity))

    # ------------------------------
    # 3. DISPARITY POST-PROCESSING
    # ------------------------------

    # mask the disparity s.t we have a reccomended filter range.
    maskedDisparity = f.maskDisparity(disparity)
    # cap the disparity since we know we don't really need most of the information there.
    cappedDisparity = f.capDisparity(disparity)

    # ------------------------------
    # 4. DISPARITY TO POINT CLOUDS
    # ------------------------------

    # project to a 3D colour point cloud
    # we have points and maskpoints because we generate a plane from the mask points and compare them to the points in the original disparity.
    points = f.projectDisparityTo3d(cappedDisparity, opt['max_disparity'], imgL)
    maskpoints = f.projectDisparityTo3d(maskedDisparity, opt['max_disparity'])

    # ------------------------------
    # 5. PLANE FINDING WITH RANSAC
    # ------------------------------
    planePoints = []
    normal = None
    try:
        # compute ransac which will give us the coefficents for our plane.
        normal, abc = f.RANSAC(maskpoints, opt['ransac_trials'])

        # we calculate the error distances between the points on the disparity and the plane.
        pointDifferences = f.calculatePointErrors(abc, points)

        # compute good points from the plane - using a threshold for a point limit.
        points = f.computePlanarThreshold(points,pointDifferences,opt['point_threshold'])

        # generate colour histogram from the road points
        histogram = f.calculateColourHistogram(points)

        # filter the colours in the points using the histogram
        points = f.filterPointsByHistogram(points, histogram, opt['road_color_thresh'])

        # convert 3D points back into 2d.
        planePoints = f.project3DPointsTo2DImagePoints(points)
        planePoints = np.array(planePoints, np.int32)
        planePoints = planePoints.reshape((-1,1,2))
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print ("*** print_tb:")
        traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
        print("There was an error with generating a plane:", e)

    # ------------------------------
    # 6. DRAW POINTS ON IMAGE
    # ------------------------------
    imageRoadMap = imgL.copy()
    for i in planePoints:
        imageRoadMap[i[0][1]][i[0][0]] = [0,255,0]
    images.append(("Image Road Map",imageRoadMap))

    # ------------------------------
    # 7. DRAW POINTS INTO OWN IMAGE
    # ------------------------------
    roadImage = []
    try:
        roadImage = f.generatePointsAsImage(planePoints)
    except Exception as e:
        print("There was an error with generating a road image:", e)
        roadImage = f.getBlackImage()
    images.append(("Road Image",roadImage))

    # ------------------------------
    # 8. CLEAN ROAD POINTS
    # ------------------------------

    # sanitise the road image.
    cleanedRoadImage = []
    resultingPoints = []
    try:
        cleanedRoadImage, resultingPoints = f.sanitiseRoadImage(roadImage, opt['img_size'])
    except Exception as e:
        print("There was an error with cleaning the road image:", e)
        # since an error was found, use the original, uncleaned planepoints.
        cleanedRoadImage = f.getBlackImage()
        resultingPoints = planePoints
    images.append(("Filtered Road Image",cleanedRoadImage))

    # ------------------------------
    # 9. DRAW ROAD AND NORMAL LINES
    # ------------------------------
    resulting_image = imgL
    try:
        # generate convex hull and draw it on the image.
        imgL, hull = f.drawContours(imgL, resultingPoints)
        # get center point from the hull points.
        center =  f.getCenterPoint(hull)
        # draw normal line.
        resulting_image = f.drawNormalLine(imgL, center, normal, disparity)
    except Exception as e:
        print("There was an error with generating a hull:", e)
        
    images.append(("Result",resulting_image))

    # ------------------------------
    # 10*. GENERATE IMAGE TILES
    # * This is optional but it shows the processed images.
    # ------------------------------

    img_tile = f.batchImages(images, opt['img_size'])

    if opt['loop'] == True:
        # display image results.
        cv2.imshow('Result',img_tile)
        f.handleKey(cv2, opt['pause_playback'], disparity, imgL, imgR, opt['crop_disparity'])
    
    # return the results.
    return img_tile, prev_disp, normal