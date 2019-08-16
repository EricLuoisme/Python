
import cv2
from Demo import Contour_Detect_Function
import numpy as np

cap = cv2.VideoCapture(0)
while(1):
    # get a frame
    ret, frame = cap.read()
    # show a frame
    new_frame = Contour_Detect_Function.Contour_Detect(frame)
    cv2.imshow("capture", new_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()