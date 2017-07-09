import cv2
import numpy as np


class ShapeDetect:
    def __init__(self):
        pass

    def filterByColor(self, color, frame):
        if color == 'blue':
            color = 120
        elif color == 'green':
            color = 60
        else:
            color = 30

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # define range of blue color in HSV yellow: 30, blue: 120, green: 60
        lower = np.array([color-10, 50, 0])
        upper = np.array([color+10, 255, 255])
        thresh = cv2.inRange(hsv, lower, upper)

        # convert the resized image to grayscale, blur it slightly,
        # and threshold it
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(thresh, (5, 5), 0)
        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
        return thresh
        # kernel = np.ones((5, 5), np.uint8)
        # return cv2.erode(thresh, kernel, iterations=3)

    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        if len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            shape = "rectangle"

        # if the shape is a pentagon, it will have 6 vertices
        elif len(approx) == 6:
            shape = "hexagon"

        # otherwise, we assume the shape is a circle
        else:
            shape = "circle"

        # return the name of the shape
        return shape