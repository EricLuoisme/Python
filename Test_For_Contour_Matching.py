
import cv2
import numpy as np
from Return_Contour import Return_Contour

img_1_full = cv2.imread('W_1.jpg')
cv2.imwrite('W_1_compressed.jpg', img_1_full, [cv2.IMWRITE_JPEG_QUALITY, 9])
img_2_full = cv2.imread('W_use.jpg')
cv2.imwrite('W_use_compressed.jpg', img_2_full, [cv2.IMWRITE_JPEG_QUALITY, 9])

img_1 = cv2.imread('W_1_compressed.jpg')
img_2 = cv2.imread('W_use_compressed.jpg')
cnt_1, sp_1 = Return_Contour(img_1)
cnt_2, sp_2 = Return_Contour(img_2, True)

if sp_1 is True:
    cnt_1 = cnt_1[0]
if sp_2 is True:
    cnt_2 = cnt_2[0]

# print(type(cnt_1))
# print(type(np.array(cnt_2)))

hull_1 = cv2.convexHull(cnt_1)
cv2.drawContours(img_1, [hull_1], -1, (0, 255, 0), 3)

hull_2 = cv2.convexHull(cnt_2)
cv2.drawContours(img_2, [hull_2], -1, (0, 255, 0), 3)

np.save('./octagon_standard_contour.npy', hull_2)
hull_standard = np.load('octagon_standard_contour.npy')

ret = cv2.matchShapes(hull_1, hull_standard, 1, 0.0)
print(ret)

cv2.imshow("Image_1", img_1)
cv2.imshow('Image_2', img_2)
cv2.waitKey(0)

