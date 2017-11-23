# imports, don't touch them lol
import cv2
import numpy as np
import matplotlib.pyplot as plt
# import csv
import functions as f
import datetime

default_options = {
    'crop_disparity' : False,
    'pause_playback' : False,
    'max_disparity' : 1024,
    'ransac_trials' : 600,
    'loop' : False,
    'point_threshold' : 0.02
}

def performStereoVision(imgL,imgR, previousDisparity=None, opt=default_options):
    
    # perform preprocessing.
    imgL, imgR = f.preProcessImages(imgL,imgR)
    # make image greyscale
    grayL, grayR = f.greyscale(imgL,imgR)

    # compute disparity image
    disparity = f.disparity(grayL,grayR, opt['max_disparity'], opt['crop_disparity'])

    # load previous disparity to fill in missing content.
    if previousDisparity is not None:
        disparity = f.fillDisparity(disparity, previousDisparity)

    # save the disparity and return that for the next iteration in the loop.
    previousDisparity = disparity

    # mask the disparity s.t we have a reccomended filter range.
    maskedDisparity = f.maskDisparity(disparity)

    # cap the disparity since we know we don't really need most of the information there.
    cappedDisparity = f.capDisparity(disparity)

    time_start = datetime.datetime.now()
    f.projectDisparityMultiProcessing(disparity,opt['max_disparity'], imgL)
    f.projectDisparityMultiProcessing(maskedDisparity,opt['max_disparity'])
    time_end = datetime.datetime.now()
    print(time_end - time_start)
    # project to a 3D colour point cloud (with or without colour)

    time_start = datetime.datetime.now()
    print("projecting disparity to 3D..")
    points = f.projectDisparityTo3d(disparity, opt['max_disparity'], imgL)
    maskpoints = f.projectDisparityTo3d(maskedDisparity, opt['max_disparity'])
    time_end = datetime.datetime.now()
    print(time_end - time_start)

    # assign reference image
    referenceImage = imgL

    # canny = f.performCanny(grayL)

    # write to file in an X simple ASCII X Y Z format that can be viewed in 3D
    # f.saveCoords(points, '3d_points.txt')

    # filter the points by height
    # function() that filters points

    # then here we compute ransac which will give us the coefficents for our plane.
    normal, abc = f.RANSAC(maskpoints, opt['ransac_trials'])

    # we calculate the error distances between the points on the disparity and the plane.

    pointDifferences = f.calculatePointErrors(abc, points)

    # here we allocate a threshold s.t if it is beyond this level, we discard the point.
    print("Point Threshold:", opt['point_threshold'])

    # compute good points.
    print("Thresholding the points..")
    points = f.computePlanarThreshold(points,pointDifferences,opt['point_threshold'])

    # now we sanitise the points.
    print("Calculating Histogram of Points..")
    histogram = f.calculateColourHistogram(points)

    pointColourThreshold = 20
    points = f.filterPointsByHistogram(points, histogram, pointColourThreshold)
    # ----------------------------------------

    # ● For the purposes of this assignment when a road has either curved road edges or other complexities due to the road configuration (e.g. junctions, roundabouts, road type, occlusions) report and display the road boundaries as far as possible using a polygon or an alternative pixel-wise boundary.
    print("Projecting 3D Points to 2D Image points..")

    # convert 3D points back into 2d.
    pts = f.project3DPointsTo2DImagePoints(points)
    pts = np.array(pts, np.int32)
    pts = pts.reshape((-1,1,2))
    # print(pts)
  
    # copy = imgL.copy()
    # plane_shape = f.generatePlaneShape(pts, copy)

    # cv2.imshow('left imaged',plane_shape)
    # # convert back to points.

    # pts = f.getPtsAgain(plane_shape)
    # imgL = f.detectObjects(imgL)
    # When the road surface plane are detected within a stereo image it must display a red polygon on the left (colour) image highlighting where the road plane has been detected as shown in Figure 1.
    # imgL = f.drawConvexHull(pts, imgL)
    # cv2.polylines(imgL,[pts],True,(0,255,255), 3)
    # print(pts)

    print("Drawing original points on image map..")
    for i in pts:
        # print(i)

        imgL[i[0][1]][i[0][0]] = [0,255,0]

    print("Sanitizing Road Points..")
    roadImage, ptz = f.generatePointsAsImage(pts, np.size(imgL, 0), np.size(imgL, 1))
    # _,contours,_ = cv2.findContours(roadImage,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    try:
        # generate convex hull 
        print("Generating Convex Hull..")
        hull = cv2.convexHull(ptz)

        # draw hull on image L.
        cv2.drawContours(imgL,[hull],0,(0,0,255),5)

        print("Generating Normal Arrow..")
        # get center point of the area.
        (x,y),_ = cv2.minEnclosingCircle(hull)
        center = (int(x),int(y))
        print("Center point:", center)
        
        newLine = f.getNormalVectorLine(center, normal, disparity)
        lineThickness = 2
        normalLineColor = (20,185,255)
        cv2.line(imgL, center, newLine, normalLineColor, lineThickness)
        circleHeadColour = (normalLineColor[0]+10, normalLineColor[1]+10, normalLineColor[2]+10)
        cv2.circle(imgL, newLine, 2, circleHeadColour, thickness=10, lineType=8, shift=0)

    
    except Exception as e:
        print("There was an error with generating a hull:", e)

    print("Loading Image..")
    cv2.imshow('Result',roadImage)


    if opt['loop'] == True:

        # show disparity
        cv2.imshow("disparity", disparity)
        # cv2.imshow("maskedDisp", maskedDisparity)
        # cv2.imshow("canny", canny)
        cv2.imshow('Result',imgL)
        cv2.imshow('road',roadImage)
        # foo, axarr = plt.subplots(2,2)
        # axarr[0,1].imshow(disparity)
        # axarr[0,0].imshow(maskedDisparity)
        # axarr[1,0].imshow(canny)
        # axarr[1,1].imshow(imgL)

        # foo.canvas.draw()
        f.handleKey(cv2, opt['pause_playback'], disparity, imgL, imgR, opt['crop_disparity'])
    return imgL, previousDisparity