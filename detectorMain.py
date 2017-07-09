# import the necessary packages
from shapeDetector import ShapeDetect
import imutils
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
sd = ShapeDetect()
while(1):
    # Take each frame
    _, frame = cap.read()

    blue = sd.filterByColor('blue', frame)
    green = sd.filterByColor('green', frame)
    yellow = sd.filterByColor('yellow', frame)
    thresh = [blue, green]

    cv2.imshow("blue", blue)
    cv2.imshow("green", green)
    cv2.imshow("yellow", yellow)

    # find contours in the thresholded image and initialize the
    # shape detector
    temp = []
    for img in thresh:
        cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        for i in cnts:
            if cv2.contourArea(i) > 10000:
                temp.append(i)
    cnts = temp
    # loop over the contours
    for c in cnts:
        # compute the center of the contour, then detect the name of the
        # shape using only the contour
        M = cv2.moments(c)
        ratio = 1
        cX = int((M["m10"] / (M["m00"] + 1e-7)) * ratio)
        cY = int((M["m01"] / (M["m00"] + 1e-7)) * ratio)
        shape = sd.detect(c)

        # multiply the contour (x, y)-coordinates by the resize ratio,
        # then draw the contours and the name of the shape on the image
        c = c.astype("float")
        c *= ratio
        c = c.astype("int")
        cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
        cv2.putText(frame, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2)

        # show the output image
        cv2.imshow("Image", frame)
        # cv2.waitKey(0)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()