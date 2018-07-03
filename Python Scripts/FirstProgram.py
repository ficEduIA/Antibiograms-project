import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('E:\Alex 2018 - 2do Semestre\Docencia UCB\Antibiograms project\Antibiograms pictures\Picture 1.jpg')
RGBimg=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

GrayscaleImage=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
BlurredImage = cv2.medianBlur(GrayscaleImage,9)
circles = cv2.HoughCircles(BlurredImage,cv2.HOUGH_GRADIENT,1,25,param1=50,param2=10,
                           minRadius=38,maxRadius=41)

fig=plt.figure()

fig.add_subplot(3,2,1)
plt.axis("off")
plt.imshow(img)

fig.add_subplot(3,2,2)
plt.axis("off")
plt.imshow(RGBimg)

fig.add_subplot(3,2,3)
plt.axis("off")
plt.imshow(GrayscaleImage, cmap='gray')

fig.add_subplot(3,2,4)
plt.axis("off")
plt.imshow(BlurredImage, cmap='gray')

auxImg=RGBimg

##circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(auxImg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(auxImg,(i[0],i[1]),2,(0,0,255),3)
for i in circles[0,:]:
    x=i[0]
    y=i[1]
    
fig.add_subplot(3,2,5)
plt.axis("off")
plt.imshow(auxImg)

plt.show()

####cv2.imshow('OriginalImage',img)
##cv2.imshow('GraysacleImage',GrayscaleImage)
##cv2.imshow('BlurredImage',GrayscaleImage)
##k = cv2.waitKey(0)
##if k == 27:         # wait for ESC key to exit
##    cv2.destroyAllWindows()

##fig, ax = plt.subplots()
##t = np.arange(0.0, iterations, 1.0)
##ax.set(xlabel='iterations', ylabel='Cost',
##       title='LOGISTIC REGRESION')
##ax.plot(t, J)
##ax.grid()
##fig.savefig("logistec.png")
##plt.show()
