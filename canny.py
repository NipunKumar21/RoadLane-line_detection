# real time webcam canny edge detection
# import cv2
# import numpy as np
# cap = cv2.VideoCapture(0)
# while(1):
#     ret, frame = cap.read()
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
#     lower_red = np.array([30,150,50])
#     upper_red = np.array([255,255,180])
#     mask = cv2.inRange(hsv, lower_red, upper_red)
#     res = cv2.bitwise_and(frame,frame, mask= mask)
#     cv2.imshow('Original',frame)
#     edges = cv2.Canny(frame,100,200)
#     cv2.imshow('Edges',edges)
#     k = cv2.waitKey(5) & 0xFF
#     if k == 27:
#         break
# cap.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("Screenshot 2023-05-04 120544.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
canny = cv2.Canny(img, 100, 200)

titles = ['image', 'canny']
images = [img, canny]
for i in range(2):
    plt.subplot(1, 2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()