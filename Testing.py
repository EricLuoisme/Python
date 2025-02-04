import numpy as np
import cv2

img = cv2.imread('W_use.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # save image with lower quality—smaller file size
cv2.imwrite('W_compare_compressed.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 9])

# read the compressed image
img = cv2.imread('W_compare_compressed.jpg')
# convert the colored image into gray one
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('gray', gray)

# Use Sobel kernel to find the contours
gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

# subtract the y-gradient from the x-gradient
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)

# blur and threshold the image
blurred = cv2.blur(gradient, (20, 20))
(_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)

cv2.imshow('blur', blurred)

# make the image closed
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

cv2.imshow('closed', closed)

# perform a series of erosions and dilations
closed = cv2.erode(closed, None, iterations=4)
closed = cv2.dilate(closed, None, iterations=4)

# find contour
contours, hierarchy= cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# if len(contours) > 1:
#     cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]
# else:
#     [cnt] = contours

# epsilon = 0.0001*cv2.arcLength(cnt,True)
# approx = cv2.approxPolyDP(cnt,epsilon,True)
# cv2.drawContours(img, [approx], -1, (0, 255, 0), 3)

# hull = cv2.convexHull(cnt, returnPoints=False)
for cnt in contours:
    # print(type(cnt))
    # print(cnt)
    hull = cv2.convexHull(cnt)
    cv2.drawContours(img, [hull], -1, (0, 255, 0), 3)
    # epsilon = 0.0001 * cv2.arcLength(cnt, True)
    # approx = cv2.approxPolyDP(cnt,epsilon,True)
    # cv2.drawContours(img, [approx], -1, (0, 255, 0), 3)

cv2.imshow('final', img)
cv2.waitKey(0)

# height, width, channels = img.shape
# for cnt in contours:
    # # use convex to approximate the contour
    # cnt_length = cv2.arcLength(cnt, True)
    # if cnt_length > 0.5 * width or cnt_length > 0.5 * height or cnt_length < .1 * height:
    #     # this should only be used for contour fitering
    #     continue
    # else:
    # epsilon = 0.0001*cv2.arcLength(cnt,True)
    # approx = cv2.approxPolyDP(cnt,epsilon,True)
    # cv2.drawContours(img, [approx], -1, (0, 255, 0), 3)
    # hull = cv2.convexHull(cnt)
    # cv2.drawContours(img, [hull], -1, (0, 255, 0), 3)


# defects = cv2.convexityDefects(cnt, hull)
#
# for i in range(defects.shape[0]):
#     s,e,f,d = defects[i,0]
#     start = tuple(cnt[s][0])
#     end = tuple(cnt[e][0])
#     far = tuple(cnt[f][0])
#     cv2.line(img,start,end,[0,255,0],2)
#     cv2.circle(img,far,5,[0,0,255],-1)

# hull = cv2.convexHull(cnt)
# cv2.drawContours(img, [hull], -1, (0, 255, 0), 3)


# mask = np.zeros(img.shape[:2],np.uint8)
# cv2.drawContours(mask, [hull], -1, 255, -1)
# dst = cv2.bitwise_and(img, img, mask=mask)

# cv2.imshow('contours', img)
# cv2.waitKey(0)


# # crop
# # (y, x) = np.where(mask == 255)
# y,x,b = np.where(mask == 255)
# (topy, topx) = (np.min(y), np.min(x))
# (bottomy, bottomx) = (np.max(y), np.max(x))
# out = out[topy:bottomy + 1, topx:bottomx + 1]

# cv2.imshow("contours", out)
# cv2.waitKey(0)

# # the code below is using smallest rectangle to root out
#
# # compute the rotated bounding box of the largest contour
# rect = cv2.minAreaRect(cnt)
# box = np.int0(cv2.boxPoints(rect))
#
# # draw a bounding box arounded the detected barcode and display the image
# cv2.drawContours(img, [box], -1, (0, 255, 0), 3)
# cv2.imshow("Image", img)
# cv2.imwrite("contoursImage2.jpg", img)
# cv2.waitKey(0)

