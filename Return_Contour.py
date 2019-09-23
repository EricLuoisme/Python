import cv2

def Return_Contour(img):

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
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 100)) # (100, 100) for samples, (20,20) for videos
    # 20 * 20 is about how large matrix that we used to combine unconnected area together
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # perform a series of erosions and dilations
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)

    # draw contour
    contours, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 1:
        cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    else:
        cnt = contours

    return cnt