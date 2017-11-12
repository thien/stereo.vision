import random
import cv2

# fixed camera parameters for this stereo setup (from calibration)

# focal length in pixels
camera_focal_length_px = 399.9745178222656
# focal length in metres (4.8 mm) 
camera_focal_length_m = 4.8 / 1000
# camera baseline in metres    
stereo_camera_baseline_m = 0.2090607502

image_centre_h = 262.0;
image_centre_w = 474.5;
    

## project_disparity_to_3d : project a given disparity image
## (uncropped, unscaled) to a set of 3D points with optional colour

def project_disparity_to_3d(disparity, max_disparity, rgb=[]):
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
                # add to points
                if(rgb.size > 0):
                    points.append([X,Y,Z,rgb[y,x,2], rgb[y,x,1],rgb[y,x,0]]);
                else:
                    points.append([X,Y,Z]);
    return points;

#####################################################################

# project a set of 3D points back the 2D image domain

def project_3D_points_to_2D_image_points(points):
    points2 = [];
    # calc. Zmax as per above
    Zmax = (camera_focal_length_px * stereo_camera_baseline_m) / 2;
    for i1 in range(len(points)):
        # reverse earlier projection for X and Y to get x and y again
        x = ((points[i1][0] * camera_focal_length_px) / Zmax) + image_centre_w;
        y = ((points[i1][1] * camera_focal_length_px) / Zmax) + image_centre_h;
        points2.append([x,y]);
    return points2;

#####################################################################

def computePlanarFitting(points):
    # https://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points
    # points = np.array(....) ... of 3D points
    # Calculating the equation of a plane from 3 points in 3D: http://mathworld.wolfram.com/Plane.html
    # [Further hint: for this assignment this can be done in full projected floating-point 3D space (X,Y,Z) or in integer image space (x,y,disparity) – see provided hints python file]
    # ....
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

#####################################################################

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)