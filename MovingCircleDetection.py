# Thomas Zanatta
import cv2
import numpy as np

video = cv2.VideoCapture("/Users/pl250661/Downloads/IMG_3529.mov")

def find_circles(vid):
    # make video gray
    gray = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    # apply edge detection to gray video
    edges = cv2.Canny(gray, 300, 400, apertureSize=3)

    # create mask to put on gray Canny video
    mask = np.zeros(vid.shape[:2], dtype="uint8")
    # draw rectangle mask around can to not mask it out
    cv2.rectangle(mask, (0, 100), (1060, 500), 255, -1)
    masked = cv2.bitwise_and(edges, edges, mask=mask)

    # defines circles to be drawn
    circles = cv2.HoughCircles(masked, cv2.HOUGH_GRADIENT, dp=1, minDist=200, param1=60, param2=45, minRadius=1, maxRadius=300)

    # if there are circles, draw them
    if circles is not None:
        circles = np.uint16(circles)
        for x in circles[0, :]:
            cv2.circle(vid, (x[0], x[1]), x[2], (255, 0, 0), 4) # circle
            cv2.circle(vid, (x[0], x[1]), 2, (0, 0, 255), 3) # center point
    return vid

if not video.isOpened():
    print("Video file could not be opened.")

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        print("Couldn't receive frame/video stopped.")
        break

    cv2.imshow("Circle Detection", find_circles(frame))
    if cv2.waitKey(1) == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
