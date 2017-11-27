<!-- 
# https://stackoverflow.com/questions/24814941/concave-hull-with-missing-edges
# https://github.com/pmneila/morphsnakes
# http://pathfinder.engin.umich.edu/documents/Feng&Taguchi&Kamat.ICRA.2014.pdf
# http://web.ipac.caltech.edu/staff/fmasci/home/astro_refs/HoughTrans_lines_09.pdf
# https://stackoverflow.com/questions/18255958/harris-corner-detection-and-localization-in-opencv-with-python
# https://stackoverflow.com/questions/32609098/how-to-fast-change-image-brightness-with-python-opencv
 -->

# Report
# cmkv68

## 1. Pre Filtering
When both images are loaded, they are faced with gamma corrections and are then converted to greyscale. The greyscale images are then histogram equalised to counter any defects on colours.

## 2. Disparity Processing

The left and right image channels are then used to create the disparity. In the event that there is information missing, two methods were tested.

### 2.1 Mean Processing
For each row in the disparity image, we calculate its non-zero mean. This has its own issues where it may seem imprecise; [expand here]

### 2.2 Previous Disparity Filling
Every black point in the disparity image is filled using the values from the previous disparity, by overlaying. This improves over time and works especially well if the car is not travelling fast.

## 3. Disparity Post-Processing
We can use a heuristic that the majority of the road can fit within a certain parameter of the image. we use a mask that fits the the road as our guide, and make a new disparity image from such.

## 4. Disparity to Point Clouds
We convert the disparities (both the original disparity, and the masked disparity) to point clouds. However, to improve the performance, we increment the step counter by 2 (such that it skips a point per iteration). For the original disparity, we also store rgb values of image_L.

This reduces the number of computations by (height * width)^2 whilst retaining sufficent points needed to compute an accurate plane. Other step counters have been considered, such as 3 or 4 but during experiments it is found to lose too much information.

The ZMax cap is no longer used (from the original code), but we use the disparity value of a given point to calculate the Z Position:

`Z = f*B/disparity(y,x).`

## 5. Plane Finding with RANSAC

Using the point cloud of the masked disparity, we use RANSAC to choose random points within them (400 points) and to compute a plane. We iterate this 600 times, storing the best plane throughout the computation. We compare its performance by calculating the mean error from our random sample of points from the masked disparity point cloud.

Computing the plane using the disparity image has been trialed (to bypass computing a 3d point cloud), but this has shown to be less precise due to the disparity range. Increasing the max_disparity does not improve this. 


## 6. Generating Road Points

For each point in the original disparity their distance from the plane using the plane coefficents. We threshold points if they are far enough.

Then, we calculate a histogram for the remaining points, using the HSV Hue value of each point. With this histogram, we remove points that are not in the most populous colours using a colour threshold.

The remaining points from the cloud are projected back to 2D image points.

## 7. Cleaning Road Points

From the Road Image, we perform a series of image manipulation operations:
- Morphological closing to fill small holes within the road image
- Eroding with a 9x9 kernel
- Capping the road image view by masking it upon a road threshold mask
- Performing another morphological closing
- Removing noisy/small pixels again through contour detection

## 8. Object Detection

With Object detection, we simply look at the points within the convex hull of the remaining road image. Points that are not recognised as the road but are within the confines of the hull are treated as objects on the road.

This is performed by:
- creating a convex hull of the image
- creating a mask using the convex hull
- inverting the original road image to show the object spots
- colouring the corresponding points yellow
- overlaying the object points onto the resulting image

## 9. Drawing Road and Normal Shapes

First, the convex hull is drawn on the image.
Then, the center point is calculated from the points generated from the convex hull.

Then, a normal vector line is generated using the center point, converting it to a 3d point, and using the normal vector, we calculate a following point by adjusting the Y point, keeping the X point and using the vector equation to calculate the new Z point.
The new 3d point is then converted back to a 2D point and the pair is sent back to be drawn on the image.

## Performance


