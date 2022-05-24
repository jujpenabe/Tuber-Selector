import math

import cv2
import numpy as np
import matplotlib.pyplot as plt

import Preprocessing.Thresholding as prp

# Display images
def show(image, title):
    plt.figure(figsize=(7, 7))
    plt.title(title)
    plt.imshow(image)
    plt.show()


def segment(file):
    listImg = []
    image = cv2.imread(file)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original = image
    # Low pass filter (blur)
    image = cv2.blur(image, (31, 31), 0)
    # To Gray
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply otsu binarization
    _, thr = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Mask original image with the OTSU binarization
    image = cv2.bitwise_and(original, original, mask=thr)

    listImg.append(thr)
    listImg.append(image)

    return listImg


def getROI(image, mask): 
    # Find contours 
    contours, hierachy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # OpenCV3 
    # Deep copies of the input image to draw results:
    polyRectImage = image.copy()
    # final ROI:
    croppedImg = 0
    # Moments
    # cnt = contours[0]
    # M = cv2.moments(cnt)
    # print( M )
    # 
    # area = cv2.contourArea(cnt)
    # print("Area: " + str(area))
    # print("Area (from moments): " + str(M['m00']))
    #Look for the outer bounding boxes:
    for i, c in enumerate(contours):
        if hierachy[0][i][3] == -1:
            # Get contour area:
            contourArea = cv2.contourArea(c)
            # Approximate the contour to a polygon:
            contoursPoly = cv2.approxPolyDP(c, 3, True)
            # Convert the polygon to a bounding rectangle:
            boundRect = cv2.boundingRect(contoursPoly)
            minRect = cv2.minAreaRect(contoursPoly)
            (x,y), radius = cv2.minEnclosingCircle(contoursPoly)
            box = cv2.boxPoints(minRect)
            box = np.int0(box)
            center = (int(x), int(y))
            radius = int(radius)
            # Set the rectangle dimensions:
            rectangleX = boundRect[0]
            rectangleY = boundRect[1]
            rectangleWidth = boundRect[0] + boundRect[2]
            rectangleHeight = boundRect[1] + boundRect[3]
            
            # Draw the rectangle:
            cv2.rectangle(polyRectImage, (int(rectangleX), int(rectangleY)), (int(rectangleWidth), int(rectangleHeight)), (0, 255, 0), 5)
            cv2.drawContours(polyRectImage, [box], 0, (255,0,0),5)
            cv2.circle(polyRectImage, center, radius, (0,0,255), 5)
            
            # Get Roundness
            # print("minRect data: " + str(minRect))
            # print("boundRect data: " +str(boundRect))
            minRectArea = minRect[1][0] * minRect[1][1]
            straightArea = boundRect[2] * boundRect[3]
            minCircleArea = radius * radius * math.pi
            
            #Crop the ROI:
            croppedImg = image[rectangleY:rectangleHeight, rectangleX:rectangleWidth]
            #croppedImg = segmented[rectangleY:rectangleWidth, rectangleX:rectangleHeight]      
            #show(croppedImg, "Cropped Image")
    return croppedImg, polyRectImage, contourArea, minRectArea, straightArea, minCircleArea

def test():
    file = "../Renders/blender_potato2_1_light.jpg"
    # images = prp.otsu_threshold(file)
    # images = segment(file)
    inputImage = cv2.imread(file) [:,:,::-1]
    image1 = prp.hsv_otsu_threshold(inputImage)
    image2 = prp.and_mask(inputImage, image1)
    image3, polyRectImage, contourArea, minArea, badArea, minCircleArea = getROI(image2, image1)
    # plt.imshow(images[0], cmap='gray', vmin=0, vmax=255)
    # plt.show()
    # plt.imshow(images[1], cmap='gray', vmin=0, vmax=255)
    # plt.show()
    squareRoundness = contourArea / minArea
    circleRoundness = contourArea / minCircleArea
    titles = ['Original Image', 'Hue Thresholding', 'Masked', 'Enclosing areas', 'ROI cuadrado']
    images = [inputImage, image1, image2, polyRectImage, image3]
    
    for i in range(len(images)):
        plt.subplot(3, 2, i + 1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    
    plt.text(1.5, 1, "Contour Area= {}".format(contourArea), transform=plt.gca().transAxes)
    plt.text(1.5, 0.8, "Minimum rectangle area= {}".format(int(minArea)), transform=plt.gca().transAxes)
    plt.text(1.5, 0.6, "Straight rectangle area= {}".format(badArea), transform=plt.gca().transAxes)
    plt.text(1.5, 0.4, "Square Roundness Factor= {}".format(round(squareRoundness, 3)), transform=plt.gca().transAxes)
    plt.text(1.5, 0.2, "Circle Roundness Factor= {}".format(round(circleRoundness, 3)), transform=plt.gca().transAxes)
    plt.show()

if __name__ == '__main__':
    test()
