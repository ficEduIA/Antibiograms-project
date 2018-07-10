import numpy as np
import matplotlib.pyplot as plt
import cv2

def showResults():
    fig=plt.figure()
    fig.suptitle("ANTIBIOGRAMS", fontsize=16)

    fig2=fig.add_subplot(2,2,1)
    plt.axis("off")
    fig2.title.set_text('Real Image')
    plt.imshow(RGBimg)

    fig3=fig.add_subplot(2,2,2)
    plt.axis("off")
    fig3.title.set_text('Filtered Image')
    plt.imshow(gray,cmap='gray')

    fig4=fig.add_subplot(2,2,3)
    plt.axis("off")
    fig4.title.set_text('Antibiotics Circles')
    plt.imshow(filteredImage,cmap='gray')

    # fig4=fig.add_subplot(2,2,4)
    # plt.axis("off")
    # fig4.title.set_text('Alos Circles')
    # plt.imshow(auxImg)
    plt.show()

def drawCircles(arrayX, image):
    for i in arrayX[0,:]:
        # draw the outer circle
        cv2.circle(image,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(image,(i[0],i[1]),2,(0,0,255),3)


img = cv2.imread('..\Antibiograms pictures\Picture 3.jpg')
RGBimg=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# auxImg=RGBimg.copy()###Esta linea puede estar de mas
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
gray = cv2.GaussianBlur(gray,(3,3),0)
gray = cv2.medianBlur(gray,9)
gray = cv2.equalizeHist(gray)
filteredImage = cv2.equalizeHist(gray)


s,l=filteredImage.shape[:2]
for i in range(s):
    for j in range(l):
        if filteredImage[i,j]>127:
            filteredImage[i,j]=255
        else:
            filteredImage[i,j]=0

# erosio and dilation
kernel = np.ones((4,4),np.uint8)
filteredImage = cv2.erode(filteredImage,kernel,iterations = 1)
filteredImage = cv2.dilate(filteredImage,kernel,iterations = 1)





# # antibioticCircles = cv2.HoughCircles(filteredImage,cv2.HOUGH_GRADIENT,1,50,param1=25,param2=10,
# #                            minRadius=4,maxRadius=10)
# # antibioticCircles = np.uint16(np.around(antibioticCircles))
# drawCircles(antibioticCircles,auxImg)
showResults()
