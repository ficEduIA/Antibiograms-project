import numpy as np
import cv2

img = cv2.imread('E:\Alex 2018 - 2do Semestre\Docencia UCB\Antibiograms project\Antibiograms pictures\Picture 1.jpg')
BWimage=cv2
cv2.imshow('image',img)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('messigray.png',img)
    cv2.destroyAllWindows()
