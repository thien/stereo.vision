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
    'road_color_thresh': 10,        # remove points from roadpts if it isn't in the x most populous colours 
    'loop': True,
    'point_threshold' : 0.05,
    'image_tiles' : True,           # show all images involved in the process or not
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

    # add the resulting image to the list of images.
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
    # 6. DRAW POINTS INTO OWN IMAGE
    # ------------------------------

    imageRoadMap = imgL.copy()
    for i in planePoints:
        imageRoadMap[i[0][1]][i[0][0]] = [0,255,0]
    
    # add the resulting image to the list of images.
    images.append(("Image Road Map",imageRoadMap))

    roadImage = []
    try:
        roadImage = f.generatePointsAsImage(planePoints)
    except Exception as e:
        print("There was an error with generating a road image:", e)
        roadImage = f.getBlackImage()
    
    # add the resulting image to the list of images.
    images.append(("Road Image",roadImage))

    # ------------------------------
    # 7. CLEAN ROAD POINTS
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
    
    # add the resulting image to the list of images.
    images.append(("Filtered Road Image",cleanedRoadImage))

    # ------------------------------
    # 8. DETECT OBJECTS
    # ------------------------------

    try:
        # convex hull on the road image
        roadHull = cv2.convexHull(resultingPoints)
        # on our image, we will fill in our convex hull to make a mask.
        cleanedRoadImage2 = cleanedRoadImage.copy()
        cleanedRoadImage3 = cleanedRoadImage.copy()
        hullMask = cv2.drawContours(cleanedRoadImage2,[roadHull],0,255,-100)
        # make an inverse of our road image to show the non road objects as white.
        objectImage = cv2.bitwise_not(cleanedRoadImage3)
        # mask this image with the hull mask.
        objectImage = cv2.bitwise_and(objectImage, objectImage, mask=hullMask)
        # convert object image to bgr
        objectImage = cv2.cvtColor(objectImage, cv2.COLOR_GRAY2BGR)
        # turn image yellow.
        objectImage[np.where((objectImage == [255,255,255]).all(axis = 2))] = [0,255,255]
        # we then overlay the object image to the display image so that we can see where objects are.
        alpha = 0.4
        imgL = cv2.addWeighted(objectImage, alpha, imgL, 1 - alpha, 0, imgL)
    except Exception as e:
        print("There was an error in detecting objects:", e)

    # ------------------------------
    # 9. DRAW ROAD AND NORMAL LINES
    # ------------------------------
    resulting_image = imgL
    
    try:
        # generate convex hull and draw it on the image.
        imgL, hull = f.drawRoadLine(imgL, resultingPoints)
        # get center point from the hull points.
        center =  f.getCenterPoint(hull)
        # draw normal line.
        resulting_image = f.drawNormalLine(imgL, center, normal, disparity)
    except Exception as e:
        print("There was an error with generating a hull:", e)
        
    # add the resulting image to the list of images.
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