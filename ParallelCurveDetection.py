# Thomas Zanatta
"""
1. Import OpenCV.
2. Import Numpy.
3. Import Scipy.

4. Define "video" variable for the camera.
5. Define "find_curves" function, taking the video as a parameter.
    - Turn the video to grayscale with cvtColor.
    - Gaussian blur the gray video with GaussianBlur.
    - Apply Canny edge detection to the gray, blurred video.
    - Detect all the contours of the Canny, gray, blurred video.

    - Create a list of x values for the contour points.
    - Create a list of y values for the contour points.
    - Iterate through the contour points until you reach the coordinates.
        - Append each x value to the list.
        - Append each y value to the list.
        - Draw the point on the image.

    - Create a 4th degree polynomial to fit all the x and y values (the points) with scipy.
    - Create a graph line from this polynomial with poly1d.
    - Create random x points for this line to graph with.
    - Transpose these x points and their y values into a set of points.
    - Draw this line on the image with polylines.

    - Return the video.
6. Read the video frame, and if it is not available, notify the user.
7. Call the find_curves() function.
8. Display the window.
9. If the Q key is pressed, close the window.
"""
    
import cv2
import numpy as np
from scipy.interpolate import interp1d

video = cv2.VideoCapture(0)

def find_curves(vid):
   gray = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
   blur = cv2.GaussianBlur(gray, (5, 5), 0)
   edges = cv2.Canny(blur, 10, 200, apertureSize=3)
   contours, h = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

   xlist = []
   ylist = []
   for x in contours:
       for y in x:
           for z in y:
               x_value = int(z[0])
               y_value = int(z[1])
               xlist.append(x_value)
               ylist.append(y_value)
               cv2.circle(vid, (x_value, y_value), 3, (0, 0, 255), -1)

   f = np.polyfit(xlist, ylist, 4)
   func = np.poly1d(f)
   xnew = np.linspace(-1000, 1000, 1000)
   pts = np.array([np.transpose([xnew, func(xnew)])], dtype=np.int32)
   cv2.polylines(vid, pts, False, (0, 255, 0), 2)

   return vid

while True:
   ret, frame = video.read()

   if not ret:
       print("Could not receive frame.")
       break

   find_curves(frame)
   cv2.imshow("Parallel Line Detection", frame)

   if cv2.waitKey(10) == ord('q'):
       break

video.release()
cv2.destroyAllWindows()
