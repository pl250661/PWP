"""
This is an OpenCV program that aims to detect lane lines using image processing techniques using a video of a car moving through a road. Libraries that are used are cv2 and numpy. 
In order to detect the lines, the lines are separated by their angle, which is calculated using the arctangent of the slope multiplied by 180 and divided by π.
They are then sorted by their angle and placed into two separate lists where they are then averaged out to form the two lines, and combined into one list to create the centerline.
An arrow is added at the top right of the screen to turn whenever the car turns in the video.
"""

import cv2
import numpy as np


video = cv2.VideoCapture("/Users/pl250661/Downloads/the_stream.mov")
arrow = cv2.imread("/Users/pl250661/Downloads/arrow.png", cv2.IMREAD_UNCHANGED)


def image_process(vid):
   """
   Blanket function used to detect lines using image filtering techniques and an ROI to filter out any background lines.
   """

   color = cv2.cvtColor(vid, cv2.COLOR_HSV2RGB)
   contrast = cv2.convertScaleAbs(color, alpha=2, beta=0)
   blur = cv2.GaussianBlur(contrast, (13, 13), 0)
   edges = cv2.Canny(blur, 1, 40, apertureSize=3)


   ROI_bottom_left = (200, 1000)
   ROI_bottom_right = (1600, 1000)
   ROI_top_left = (700, 600)
   ROI_top_right = (1100, 600)
   points = np.array([[ROI_bottom_left], [ROI_bottom_right], [ROI_top_right], [ROI_top_left]])
   points = points.reshape((-1, 1, 2))


   mask = np.zeros((vid.shape[:2]), np.uint8)
   cv2.fillPoly(mask, [points], 255)
   ROI = cv2.bitwise_and(edges, edges, mask=mask)


   lines = cv2.HoughLinesP(ROI, 1, np.pi / 180, 20, minLineLength=90, maxLineGap=100)
   left_lines = []
   right_lines = []
   both_lines = []


   if lines is not None:
       for line in lines:
           x1, y1, x2, y2 = line[0]
           slope = (y2 - y1) / (x2 - x1 + 0.00001)
           angle = np.arctan(slope) * 180 / np.pi


           if angle > 5 and abs(angle) > 25:
               left_lines.append(line)
           elif angle < 5 and abs(angle) > 25:
               right_lines.append(line)


   left_lines_average = np.average(left_lines, axis=0)
   if len(left_lines) > 0:
       both_lines.append(left_lines_average)


   right_lines_average = np.average(right_lines, axis=0)
   if len(right_lines) > 0:
       both_lines.append(right_lines_average)


   if len(both_lines) == 2:
       for x in both_lines:
           for line in x:
               x1 = int(line[0])
               y1 = int(line[1])
               x2 = int(line[2])
               y2 = int(line[3])
               slope = (y2 - y1) / (x2 - x1 + 0.0001)
               cv2.line(vid, (x1, y1), (x2, y2), (255, 0, 0), 3)
               print(f"Line: y = {slope}x + {y1 - slope * x1}")


       if len(both_lines) > 0:
           center_line = np.average(both_lines, axis=0)


           center_x1 = int(center_line[0][0])
           center_y1 = int(center_line[0][1])
           center_x2 = int(center_line[0][2])
           mid_x = (center_x1 + center_x2) // 2
           center_y2 = vid.shape[0]
           center_slope = (center_x2 - center_x1) / (center_y2 - center_y1 + 0.000001)
           cv2.line(vid, (mid_x, center_y1), (mid_x, center_y2), (0, 255, 0), 3)
           print(f"Center line: y = {center_slope}x + {center_y1 - center_slope * center_x1}")


   return vid


width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))


arrow_width = 70
arrow_height = 100

while True:
   """
   Loop used to display the arrow on the image.
   """
   
   ret, frame = video.read()
   if not ret:
       break

   x_offset = 1800
   y_offset = 50

   arrow_resized = cv2.resize(arrow, (arrow_width, arrow_height))
   
   region_x_end = x_offset + arrow_resized.shape[1]
   region_y_end = y_offset + arrow_resized.shape[0]
   
   if region_x_end > frame.shape[1]:
        x_offset = frame.shape[1] - arrow_resized.shape[1]
        
   if region_y_end > frame.shape[0]:
        y_offset = frame.shape[0] - arrow_resized.shape[0]
   
   region = frame[y_offset:y_offset + arrow_resized.shape[0], x_offset:x_offset + arrow_resized.shape[1]]
   
   if arrow_resized.shape[2] == 4:
       arrow_rgb = arrow_resized[:, :, :3]
       arrow_alpha = arrow_resized[:, :, 3] / 255.0
   
       for x in range(0, 3):
           region[:, :, x] = (arrow_alpha * arrow_rgb[:, :, x] + (1 - arrow_alpha) * region[:, :, x])
   else:
       frame[y_offset:y_offset + arrow_resized.shape[0], x_offset:x_offset + arrow_resized.shape[1]] = arrow_resized

   cv2.imshow("image processing", image_process(frame))


   if cv2.waitKey(10) == ord('q'):
       break


video.release()
cv2.destroyAllWindows()
