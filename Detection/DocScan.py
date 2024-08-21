import cv2
import numpy as np
from detection import utlis

heightImg = 640
widthImg  = 480

def stack_img_generator(img):

    imgWarpColored = None

    img = cv2.resize(img, (widthImg, heightImg)) 
    imgBlank = np.zeros((heightImg,widthImg, 3), np.uint8) 
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) 
    imgThreshold = cv2.Canny(imgBlur,50,150)
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2) 
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1) 

    imgContours = img.copy()
    imgBigContour = img.copy() 
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10) 


    biggest, maxArea = utlis.biggestContour(contours) 

    if biggest.size != 0:
        biggest=utlis.reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20) 
        imgBigContour = utlis.drawRectangle(imgBigContour,biggest,2)
        pts1 = np.float32(biggest) 
        pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) 
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

        imgWarpColored=imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
        imgWarpColored = cv2.resize(imgWarpColored,(widthImg,heightImg))

        imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
        imgAdaptiveThre= cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
        imgAdaptiveThre=cv2.medianBlur(imgAdaptiveThre,3)

        imageArray = ([img,imgGray,imgThreshold,imgContours],
                        [imgBigContour,imgWarpColored, imgWarpGray,imgAdaptiveThre])

    else:
        imageArray = ([img,imgGray,imgThreshold,imgContours],
                        [imgBlank, imgBlank, imgBlank, imgBlank])

    lables = [["Original","Gray","Threshold","Contours"],
                ["Biggest Contour","Warp Prespective","Warp Gray","Adaptive Threshold"]]

    stackedImage = utlis.stackImages(imageArray,0.75,lables)
    
    if imgWarpColored is not None:
        return stackedImage, imgWarpColored
    else:
        return stackedImage, None