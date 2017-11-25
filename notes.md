- Will need to perform ransac against the disparity instead of the coordinates.
    tried, but accuracy was poor and you'll need to convert it to 3d points anyway.
- will need to draw the normal direction.
    done.
- will need to parallelise the 3d conversion.
    no longer necessary.
- will need to display multiple images in a set.
    done.
- set up video recording
    done.
- verify that it runs on durham Computers
    done.
- print normal properly
    done.
- filter by HSV.
    done.
- fix normal arrow.
    done.
- detect objects that obstruct the plane.
    done but not very well.

---

The 3D (localization) uncertainty is caused by a low
accuracy computation of the disparity value and is mainly
visible in the depth value. The height is also influenced and
modeling the uncertainty will help for a robust detection of
the road. 

---

get the convex hull of the resulting road plane (post filter)
fill it in and use that as a mask
mask against our current road spots
look for black spots within that filter
we know they're objects