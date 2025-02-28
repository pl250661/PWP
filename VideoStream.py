import cv2
import numpy as np


video = cv2.VideoCapture("/Users/pl250661/Downloads/the_stream.mp4")
arrow = cv2.imread("/Users/pl250661/Downloads/arrow.png", cv2.IMREAD_UNCHANGED)




def image_process(vid):
   color = cv2.cvtColor(vid, cv2.COLOR_HLS2RGB)
   contrast = cv2.convertScaleAbs(color, alpha=1.1, beta=1.1)
   blur = cv2.GaussianBlur(contrast, (17, 17), 0)


   edges = cv2.Canny(blur, 1, 20, apertureSize=3)


   ROI_bottom_left = (200, 1000)
   ROI_bottom_right = (1600, 1000)
   ROI_top_left = (700, 650)
   ROI_top_right = (1100, 650)
   points = np.array([[ROI_bottom_left], [ROI_bottom_right], [ROI_top_right], [ROI_top_left]])
   points = points.reshape((-1, 1, 2))


   mask = np.zeros((vid.shape[:2]), np.uint8)
   cv2.fillPoly(mask, [points], 255)
   ROI = cv2.bitwise_and(edges, edges, mask=mask)


   lines = cv2.HoughLinesP(ROI, 1, np.pi / 180, 70, minLineLength=90, maxLineGap=80)


   if lines is not None:
       for line in lines:
           x1, y1, x2, y2 = line[0]
           slope = (y2 - y1) / (x2 - x1 + 0.00001)
           angle = np.arctan(slope) * 180 / np.pi
           if abs(angle) > 25:
               cv2.line(vid, (x1, y1), (x2, y2), (255, 0, 0), 2)


       return vid




width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))


arrow_width = 70
arrow_height = int((arrow_width / arrow.shape[1]) * arrow.shape[0])
arrow_resized = cv2.resize(arrow, (arrow_width, arrow_height))


while True:
   ret, frame = video.read()
   if not ret:
       break


   x_offset = 50
   y_offset = 50


   region = frame[y_offset:y_offset + arrow_resized.shape[0], x_offset:x_offset + arrow_resized.shape[1]]


   if arrow_resized.shape[2] == 4:
       arrow_rgb = arrow_resized[:, :, :3]
       arrow_alpha = arrow_resized[:, :, 3] / 255.0


       for c in range(0, 3):
           region[:, :, c] = (arrow_alpha * arrow_rgb[:, :, c] +
                              (1 - arrow_alpha) * region[:, :, c])
   else:
       frame[y_offset:y_offset + arrow_resized.shape[0], x_offset:x_offset + arrow_resized.shape[1]] = arrow_resized


   cv2.imshow("image processing", image_process(frame))


   if cv2.waitKey(10) == ord('q'):
       break


video.release()
cv2.destroyAllWindows()
