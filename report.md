<!-- 

● Report:
○ Discussion / detail of solution design and choices made 5%
○ Statistical evidence of the performance of system at the task 5%
  -->

# Report
# cmkv68

## 1. Pre Filtering
When both images are loaded, they are faced with gamma corrections and are then converted to greyscale. The greyscale images are then histogram equalised to counter any defects on colours.

## 2. Disparity Processing
<!-- ○ any image pre-filtering performed (or similar first stage processing) -->

The left and right image channels are then used to create the disparity.
In the event that there is information missing, two methods were tested.

During my testing, I have found that the quality of the left and right image channels improve the performance of the disparity.

### 2.1 Mean Processing
For each row in the disparity image, we calculate its non-zero mean. This has its own issues where it may seem imprecise; [expand here]

### 2.2 Previous Disparity Filling
Every black point in the disparity image is filled using the values from the previous disparity, by overlaying. This improves over time and works especially well if the car is not travelling fast.

## 3. Disparity Post-Processing
We can use a heuristic that the majority of the road can fit within a certain parameter of the image. we use a mask 

It is important to note that 
<!-- ○ the successful use of any heuristics to speed up processing times or reduce false
    detections (including at the stage of colour pre-filtering) -->

    <!--  ○ choice of region of interest extraction methodology -->

## 4. Disparity to Point Clouds
We convert the disparity to point clouds. However, to improve the performance, we increment the step counter by 2 (such that it skips a point per iteration). This reduces the number of computations by (height * width)^2 whilst retaining sufficent points needed to compute an accurate plane. Other step counters have been considered, such as 3 or 4 but during experiments it is found to lose too much information.

<!--    ○ the successful use of any heuristics to speed up processing times or reduce false
    detections (including at the stage of colour pre-filtering) -->

## 5. Plane Finding with RANSAC
Several approaches were taken to compute the plane:

<!--   ○ effective use of RANSAC for 3D planar surface detection 30% -->
### 5.1 Point Cloud Computation
Using the point cloud comparison.

### 5.2 Disparity Image Computation 
This has shown to be less precise due to the image range. Increasing the max_disparity does not improve this. 

## 6. Setup Points into Image

## 7. Cleaning Road Points
<!-- ○ automatically adjusting the parameters initial region of interest extraction or image prefiltering
    based on some form of preliminary analysis image
 -->

## 8. Object Detection
<!--  ○ automatically detecting and highlighting obstacles that rise above the road surface plane
    (vehicles, pedestrians, bollards etc.) as they appear directly in front of the vehicle s -->

## 9. Drawing Road and Normal Shapes
○ plotting of the planar normal direction direction glyph / vector in the image
<!--     ● General performance on detection of planar orientation and bounds in the imagery **
    (taking into account accuracy, false detection, missed detection, failures etc.) 30% -->