"""
Sujay Jain.
ECE 180 DA
Lab 1

This is the code for task 4. This code is adapted from multiple tutorials, namely:
1. https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html (How to capture video)
2. https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html (how to convert between rgb and hsv)
3. https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html (how to threshold)
4. https://docs.opencv.org/4.x/da/d0c/tutorial_bounding_rects_circles.html (how the general flow of bounding rectangle should work. Note this uses edge based contouring which I replace with rgb/hsv thresholding)
5. https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097 (how to find dominant color in image - this was adapted to account for video input and the color was based only on a central rectangle)
"""

"""For part 1:

I have two code blocks below: one for RGB and the other for HSV below.

HSV seems to be a little better experimentally. To accurately capture the object in rgb, I had to use a much larger range and it needed
more fine tuning. See screenshots titled 1_hsv and 1_rgb.


"""

# Using RGB

# import cv2
# import numpy as np

# # main function
# def track_object_by_rgb(rgb_lower, rgb_upper):
#     # Capture video from the webcam
#     cap = cv2.VideoCapture(1)

#     while True:
#         ret, frame = cap.read()

#         if not ret:
#             print("Failed to grab frame")
#             break

#         mask = cv2.inRange(frame, rgb_lower, rgb_upper)

#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         if contours:
#             largest_contour = max(contours, key=cv2.contourArea)
#             x, y, w, h = cv2.boundingRect(largest_contour)
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box using specified size; fixed one dim.

#         cv2.imshow("Object Tracking", frame)
#         # boiler plate code
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# # Found this online and modified https://www.roborealm.com/help/Color_Threshold.php.

# rgb_lower = np.array([100, 0, 0])
# rgb_upper = np.array([255, 100, 100])

# # Start object tracking using RGB values
# track_object_by_rgb(rgb_lower, rgb_upper)


# # Using HSV. Modified from above. Major difference is using tutorial 2 to convert between rgb and hsv.

# import cv2
# import numpy as np

# # track object by color + bound box
# def track_object_by_color(hsv_lower, hsv_upper):
#     cap = cv2.VideoCapture(0)  # this is finicky now. idk why 0 works sometimes and other times 1. Maybe it depends on other open apps?

#     while True:
#         ret, frame = cap.read()

#         if not ret:
#             print("Failed to grab frame")
#             break

#         hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#         mask = cv2.inRange(hsv_frame, hsv_lower, hsv_upper)

#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         if contours:
#             largest_contour = max(contours, key=cv2.contourArea)
#             x, y, w, h = cv2.boundingRect(largest_contour)
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box

#         cv2.imshow("Object Tracking", frame)
        # boiler plate exit code
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# range for color assuming blue. Range found online.
# hsv_lower = np.array([90, 50, 50])
# hsv_upper = np.array([130, 255, 255])

# track_object_by_color(hsv_lower, hsv_upper)


"""Part 2: I tested using only hsv since it is thought as better for thresholding. I did not find any meaningful difference.
Maybe I did not darken the room enough. See hsv_light vs hsv_dark"""


""" Part 3: I found that I can generally pick up blue color but anything outside blue domain is not picked up (as should be
expected based on my threshold values). Reducing my phone's brightness signficiantly reduced the accuracy of the bounding box. See hsv_blue/red_light/dark """


"""Part 4: I wrote this part of code by take the structure of the previous code and using tutorial 4 and adapted it for video.

I observe that the one on phone is signficantly more robust to brightness/darkness. However, I did this experiment under yellow light. So that may have impacted
the results. Also, phone was easier to fit in the bounding box rather than water bottle. See dom_phone/object_light/dark.
 """

import cv2
import numpy as np
from sklearn.cluster import KMeans

# hist code from tutorial 4
def find_histogram(clt):
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist

# color bar code from tutorial 4
def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    return bar

cap = cv2.VideoCapture(0)

resize_factor = 0.5  # Added because I couldn't see color bar

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    #  Added because I couldn't see color bar
    frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)

    # central rectangle
    height, width, _ = frame.shape
    rect_x = width // 4
    rect_y = height // 4
    rect_w = width // 2
    rect_h = height // 2
    roi = frame[rect_y:rect_y + rect_h, rect_x:rect_x + rect_w]

    roi = roi.reshape((roi.shape[0] * roi.shape[1], 3))

    clt = KMeans(n_clusters=3)
    clt.fit(roi)

    hist = find_histogram(clt)
    bar = plot_colors2(hist, clt.cluster_centers_)

    cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (0, 255, 0), 2)

    cv2.imshow("Video Stream", frame)

    cv2.imshow("Dominant Colors", bar)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
