import cv2
import numpy as np

video = cv2.VideoCapture("/Users/pl250661/Downloads/IMG_3475.mov")
arrow = cv2.imread("/Users/pl250661/Downloads/arrow.png", cv2.IMREAD_UNCHANGED)

def find_curves(vid):
  color = cv2.cvtColor(vid, cv2.COLOR_RGB2HLS)
  contrast = cv2.convertScaleAbs(color, alpha=2, beta=0.2)
  blur = cv2.GaussianBlur(contrast, (3, 3), 0)
  ROI = blur[500:1000, 500:1300]
  edges = cv2.Canny(ROI, 300, 400, apertureSize=5)
  # contours, h = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE, offset=(500, 500))
  lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 260, minLineLength=10, maxLineGap=100)


  # xlist = []
  # ylist = []
  # for x in contours:
  #     for y in x:
  #         for z in y:
  #             x_value = int(z[0])
  #             y_value = int(z[1])
  #             xlist.append(x_value)
  #             ylist.append(y_value)
  #             cv2.circle(vid, (x_value, y_value), 3, (0, 0, 255), -1)




  if lines is not None:
      for line in lines:
          x1, y1, x2, y2 = line[0]
          cv2.line(vid, (x1, y1), (x2, y2), (255, 0, 0), 2)

  return vid

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

arrow_width = 200
arrow_height = int((arrow_width / arrow.shape[1]) * arrow.shape[0])
arrow_resized = cv2.resize(arrow, (arrow_width, arrow_height))

output_video_path = '/Users/Ipat/Downloads/IMG_3475.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, video.get(cv2.CAP_PROP_FPS), (width, height))


while True:
   ret, frame = video.read()
   if not ret:
       break


   x_offset = 50
   y_offset = 50


   region = frame[y_offset:y_offset+arrow_resized.shape[0], x_offset:x_offset+arrow_resized.shape[1]]


   if arrow_resized.shape[2] == 4:
       arrow_rgb = arrow_resized[:, :, :3]
       arrow_alpha = arrow_resized[:, :, 3] / 255.0


       for c in range(0, 3):
           region[:, :, c] = (arrow_alpha * arrow_rgb[:, :, c] +
                           (1 - arrow_alpha) * region[:, :, c])
   else:
       frame[y_offset:y_offset+arrow_resized.shape[0], x_offset:x_offset+arrow_resized.shape[1]] = arrow_resized


   out.write(frame)


video.release()
out.release()
