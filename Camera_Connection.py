#
# import cv2
# import Contour_Detect_Function
#
# cap = cv2.VideoCapture(0)
# cap_2 = cv2.VideoCapture(1)
# while(1):
#     # get a frame
#     ret, frame = cap.read()
#     ret_2, frame_2 = cap_2.read()
#     # show a frame
#     new_frame = Contour_Detect_Function.Contour_Detect(frame)
#     new_frame_2 = Contour_Detect_Function.Contour_Detect(frame_2)
#     cv2.imshow("capture_1", new_frame)
#     cv2.imshow("capture_2", new_frame_2)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cap_2.release()
# cv2.destroyAllWindows()


import cv2
import Contour_Detect_and_Cut


cap = cv2.VideoCapture(2)
while(1):
    # get a frame
    ret, frame = cap.read()
    # show a frame
    new_frame, new_frame_2 = Contour_Detect_and_Cut.Contour_Detect(frame)

    cv2.namedWindow("capture", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("capture", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("capture", new_frame)

    cv2.namedWindow("project", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("project", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("project", new_frame_2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


