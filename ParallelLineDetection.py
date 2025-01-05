# Thomas Zanatta

import cv2
import numpy as np

video = cv2.VideoCapture(0)

def find_lines(vid):
   gray = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)

   blur = cv2.GaussianBlur(gray, (1, 1), 0)

   edges = cv2.Canny(blur, 500, 700, apertureSize=5)

   # https: //docs.opencv.org/4.x/d3/de6/tutorial_js_houghlines.html
   lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 300, minLineLength=100, maxLineGap=500)

   if lines is not None:
       angle_threshold = 10
       intercept_threshold = 300
       parallel_lines = []
       for line1 in lines:
           for line2 in lines:
               if line1 is not line2:
                   x1, y1, x2, y2 = line1[0]
                   x3, y3, x4, y4 = line2[0]

                   slope1 = (y2 - y1) / (x2 - x1 + 0.00001)
                   intercept1 = y1 - slope1 * x1
                   slope2 = (y4 - y3) / (x4 - x3 + 0.00001)
                   intercept2 = y3 - slope2 * x3

                   angle1 = np.arctan(slope1) * 180 / np.pi
                   angle2 = np.arctan(slope2) * 180 / np.pi

                   if abs(angle1-angle2) < angle_threshold and abs(intercept1 - intercept2) > intercept_threshold:
                       parallel_lines.append((line1, line2))

       for line1, line2 in parallel_lines:
           x1, y1, x2, y2 = line1[0]
           x3, y3, x4, y4 = line2[0]
           cv2.line(vid, (x1, y1), (x2, y2), (255, 0, 0), 2)
           cv2.line(vid, (x3, y3), (x4, y4), (255, 0, 0), 2)
           if line1 is not None and line2 is not None:
               center_x_1 = (x1 + x3) // 2
               center_y_1 = (y1 + y3) // 2
               center_x_2 = (x2 + x4) // 2
               center_y_2 = (y2 + y4) // 2
               cv2.line(vid, (center_x_1, center_y_1), (center_x_2, center_y_2), (0, 0, 255), 2)

   return vid

while True:
   ret, frame = video.read()

   if not ret:
       print("Could not receive frame.")
       break

   find_lines(frame)
   cv2.imshow("Parallel Line Detection", frame)

   if cv2.waitKey(10) == ord('q'):
       break

video.release()
cv2.destroyAllWindows()
 
