import cv2
import numpy as np

img = cv2.imread('FFF03A_1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# save image with lower quality—smaller file size
cv2.imwrite('FFF_compressed.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 9])

# read the compressed image
img = cv2.imread('FFF_compressed.jpg')
# convert the colored image into gray one
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
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 100)) # (100, 100) for samples, (20,20) for videos
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# cv2.imshow('1', closed)

# perform a series of erosions and dilations
closed = cv2.erode(closed, None, iterations=4)
closed = cv2.dilate(closed, None, iterations=4)

# draw contour
contours, hierarchy= cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]

# epsilon = 0.0001*cv2.arcLength(cnt,True)
# approx = cv2.approxPolyDP(cnt,epsilon,True)
# cv2.drawContours(img, [approx], -1, (0, 255, 0), 3)

hull = cv2.convexHull(cnt)
# cv2.drawContours(img, [hull], -1, (0, 255, 0), 3)

mask = np.zeros_like(img)
cv2.drawContours(mask, [hull], -1, (0, 255, 0), 3)
out = img.copy()  # Extract out the object and place into output image
out[mask == 255] = img[mask == 255]

# crop
# (y, x) = np.where(mask == 255)
y,x,b = np.where(mask == 255)
(topy, topx) = (np.min(y), np.min(x))
(bottomy, bottomx) = (np.max(y), np.max(x))
out = out[topy:bottomy + 1, topx:bottomx + 1]

cv2.imshow('cut', out)
cv2.waitKey(0)