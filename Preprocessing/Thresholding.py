# The next threshold functions are based on the OpenCV thresholding documentation.
# https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html

import cv2
import numpy as np
from matplotlib import pyplot as plt

def simple_threshold(image):
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

    titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']

    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

    for i in range(len(titles)):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

def adaptive_threshold(image):
    img = cv2.imread(image)[:, :, ::-1]

    blured_img = cv2.medianBlur(img, 3)
    ret, th1 = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(blured_img, 110, 255, cv2.THRESH_BINARY)
    grayscale_img = cv2.imread('potato2.jpg', 0)
    ret3, th3 = cv2.threshold(grayscale_img, 110, 255, cv2.THRESH_BINARY)
    th4 = cv2.adaptiveThreshold(grayscale_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 1)
    th5 = cv2.adaptiveThreshold(grayscale_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 1)
    titles = ['Original Image', 'Blured Image', 'G.Thresholding v=110', 'G.Thresholding v=110 Blured',
              'Grayscale Image', 'G.Thresholding v=110 Gray',
              'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images = [img, blured_img, th1, th2, grayscale_img, th3, th4, th5]
    for i in range(len(images)):
        plt.subplot(4, 2, i + 1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()
    
def hsv_otsu_threshold(image, value_min, value_max, channel = 0):
    # Convertir de formato BGR a RGB
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Aplicar filtro paso bajo (blur)
    image = cv2.blur(image,(50,50),0)
    # Convertir imágen a espacio de color HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Separar canales hsv
    hsv = cv2.split(hsv)

    th_min = (hsv[channel] >= value_min)
    th_max = (hsv[channel] <= value_max)
    th = (th_min * th_max) if value_min < value_max else (th_min + th_max)
    th = th * 255
    th = th.astype(np.uint8)
    
    # Aplicar binarización OTSU
    _, thr = cv2.threshold(th, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return hsv[channel],thr
def and_mask(original, masked):
    masked = cv2.bitwise_and(original, original, mask=masked)
    return masked
def otsu_threshold(image):
    listImg = []
    # Noisy image input
    original = image  # (Test with RGB and GBR)
    # global thresholding with color
    # ret0, th0 = cv2.threshold(original, 110, 255, cv2.THRESH_BINARY)
    # To grayscale
    # img = cv2.cvtColor(th0, cv2.COLOR_BGR2GRAY)
    # Otsu's thresholding
    # ret1, th1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    ret, th2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = cv2.bitwise_and(original, original, mask=th2)
    
    listImg.append(th2)
    listImg.append(img)
    # Return filtered images
    return listImg;
 
if __name__ == '__main__':
    print("... Running Thresholding Directly...")
    # threshold()
    # adaptive_threshold()
    otsu_threshold('../Samples/Renders/blender_potato1.jpg')