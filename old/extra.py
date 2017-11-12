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