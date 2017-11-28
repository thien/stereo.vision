import cv2
import numpy as np
import functions as f
import time
import sys
import traceback
import csv

# default values for stereo vision operations
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
    'record_stats' : False,
    'video_filename' : 'previous.avi'
}

def performStereoVision(imgL,imgR, prev_disp=None, opt=default_opts):
    if 'frame' not in opt:
        opt['frame'] = 1
    else:
        opt['frame'] += 1
    # initiate stats list.
    stats = {}
    stats["Frame"] = opt['frame']
    # initiate images list.
    images = []
    # add start timer.
    start_time = time.time()

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
        stats["Planar Points Before"] = len(points)
        # generate colour histogram from the road points
        histogram = f.calculateColourHistogram(points)

        # filter the colours in the points using the histogram
        points = f.filterPointsByHistogram(points, histogram, opt['road_color_thresh'])
        stats["Planar Points After"] = len(points)

        stats["Planar Pre-Filtering Accuracy"] =  stats["Planar Points After"]/stats["Planar Points Before"]
        # convert 3D points back into 2d.
        planePoints = f.project3DPointsTo2DImagePoints(points)
        planePoints = np.array(planePoints, np.int32)
        planePoints = planePoints.reshape((-1,1,2))

        # add to stats that we computed a plane properly.
        stats["Computed Planar"] = 1
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print ("*** print_tb:")
        traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
        print("There was an error with generating a plane:", e)
        stats["Planar Points Before"] = "-"
        stats["Planar Points After"] = "-"
        stats["Planar Pre-Filtering Accuracy"] = "-"
        stats["Computed Planar"] = 0

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
    # 8. DETECT OBSTACLES
    # ------------------------------

    try:
        # convex hull on the road image
        roadHull = cv2.convexHull(resultingPoints)
        # on our image, we will fill in our convex hull to make a mask.
        cleanedRoadImage2 = cleanedRoadImage.copy()
        cleanedRoadImage3 = cleanedRoadImage.copy()
        hullMask = cv2.drawContours(cleanedRoadImage2,[roadHull],0,255,-100)
        # make an inverse of our road image to show the non road obstacles as white.
        obstacleImage = cv2.bitwise_not(cleanedRoadImage3)
        # mask this image with the hull mask.
        obstacleImage = cv2.bitwise_and(obstacleImage, obstacleImage, mask=hullMask)
        # convert obstacle image to bgr
        obstacleImage = cv2.cvtColor(obstacleImage, cv2.COLOR_GRAY2BGR)
        # turn image yellow.
        obstacleImage[np.where((obstacleImage == [255,255,255]).all(axis = 2))] = [0,255,255]
        # we then overlay the obstacle image to the display image so that we can see where obstacles are.
        alpha = 0.4
        imgL = cv2.addWeighted(obstacleImage, alpha, imgL, 1 - alpha, 0, imgL)
    except Exception as e:
        print("There was an error in detecting obstacles:", e)

    # ------------------------------
    # 9. DRAW ROAD AND NORMAL LINES
    # ------------------------------
    resulting_image = imgL
    
    try:
        # generate convex hull and draw it on the image.
        imgL, hull = f.drawRoadLine(imgL, resultingPoints)
        # get center point from the hull points.
        center =  f.getCenterPoint(hull)
        stats["Center Point X"] = center[0]
        stats["Center Point Y"] = center[1]
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

    # calculate time taken and add it to stats.
    stats["Time Taken"] = round(time.time() - start_time, 3)

    if opt['loop'] == True:
        # display image results.
        cv2.imshow('Result',img_tile)
        f.handleKey(cv2, opt['pause_playback'], disparity, imgL, imgR, opt['crop_disparity'])

    if opt['record_stats']:
        if opt['frame'] == 1:
            # write headers for first frame.
            headers = []
            for key in stats.keys():
                headers.append(str(key))
            with open("statistics.csv", 'w') as fp:
                writer = csv.writer(fp, delimiter=',')
                writer.writerow(headers)
   
        # set up input to write to csv file.
        statistics = []
        for key in stats.keys():
            statistics.append(str(stats[key]))

        # write results to file.
        with open("statistics.csv", 'a') as fp:
            writer = csv.writer(fp, delimiter=',')
            writer.writerow(statistics)

    # return the results.
    return img_tile, prev_disp, normal