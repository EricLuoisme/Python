
import cv2
import numpy as np
from Contour_Detect_Function import Contour_Detect
from Return_Contour import Return_Contour

# def Draw_Contour_Points (cnt, img):
#     hull = cv2.convexHull(cnt, returnPoints=False)
#     defects = cv2.convexityDefects(cnt, hull)
#
#     for i in range(defects.shape[0]):
#         s, e, f, d = defects[i, 0]
#         start = tuple(cnt[s][0])
#         end = tuple(cnt[e][0])
#         far = tuple(cnt[f][0])
#         cv2.line(img, start, end, [0, 255, 0], 2)
#         cv2.circle(img, far, 5, [0, 0, 255], -1)
#
#     # cv2.drawContours(img, [hull], -1, (0, 255, 0), 3)
#
#     return img


img_1 = cv2.imread('FFF03A_0_c.jpg')
img_2 = cv2.imread('Capture.PNG')

cnt_1 = Return_Contour(img_1)
cnt_2 = Return_Contour(img_2)

# all_cont_img_1 = Contour_Detect(img_1)
# all_cont_img_2 = Contour_Detect(img_2)

hull_1 = cv2.convexHull(cnt_1)
cv2.drawContours(img_1, [hull_1], -1, (0, 255, 0), 3)

hull_2 = cv2.convexHull(cnt_2)
cv2.drawContours(img_2, [hull_2], -1, (0, 255, 0), 3)

# ret = cv2.matchShapes(hull_1, hull_2, 1, 0.0)
# print(ret)
# print()
# print(type(hull_1))
# print(hull_1)

np.save('./test_saving.npy', hull_1)
hull_333 = np.load('test_saving.npy')

ret = cv2.matchShapes(hull_2, hull_333, 1, 0.0)
print(ret)

cv2.imshow("Image_1", img_1)
cv2.imshow('Image_2', img_2)
cv2.waitKey(0)

