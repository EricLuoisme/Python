
import cv2
import numpy as np

def Contour_Detect(img):

    original_img = img.copy()

    # conver colored image into gray one
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use Sobel kernel to find the contours
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    # blur and threshold the image
    blurred = cv2.blur(gradient, (20, 20))
    (_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)

    # make the image closed
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    # 20 * 20 is about how large matrix that we used to combine unconnected area together
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # perform a series of erosions and dilations
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)

    # draw contour
    contours, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 2:
        return original_img, original_img
    else:
        # cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        # hull = cv2.convexHull(cnt)
        # cv2.drawContours(img, [hull], -1, (0, 255, 0), 3)

        mask = np.zeros(original_img.shape[:2], np.uint8)
        # height, width, channels = img.shape

        for cnt in contours:
            # # use convex to approximate the contour
            # cnt_length = cv2.arcLength(cnt, True)
            # if cnt_length > 0.6 * width or cnt_length > 0.6 * height or cnt_length < 0.05 * height:
            #     continue
            # else:
            hull = cv2.convexHull(cnt)
            cv2.drawContours(img, [hull], -1, (0, 255, 0), 3)
            # remove all backgrounds
            cv2.drawContours(mask, [hull], -1, 255, -1)

            # # use epsilon to approximate the contour
            # epsilon = 0.0001 * cv2.arcLength(cnt, True)
            # approx = cv2.approxPolyDP(cnt, epsilon, True)
            # cv2.drawContours(img, [approx], -1, (0, 255, 0), 3)



        dst = cv2.bitwise_and(original_img, original_img, mask=mask)

        return img, dst