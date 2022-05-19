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

def otsu_threshold(image):
    # Noisy image input
    original = cv2.imread(image)[:, :, ::-1]  # (Test with RGB and GBR)

    # global thresholding with color
    ret0, th0 = cv2.threshold(original, 106, 255, cv2.THRESH_BINARY)

    # To grayscale
    img = cv2.cvtColor(th0, cv2.COLOR_BGR2GRAY)


    # Otsu's thresholding
    ret2, th1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret3, th2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # plot all the images and their histograms
    images = [original, 0, th0,
              img, 0, th1,
              blur, 0, th2]
    titles = ['Original Noisy Image', 'Histogram', 'G.Thresholding V=106',
              'Gray Noisy Image', 'Histogram', "Otsu's Thresholding",
              'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]
    for i in range(3):
        plt.subplot(3, 3, i * 3 + 1), plt.imshow(images[i * 3], 'gray')
        plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, i * 3 + 2), plt.hist(images[i * 3].ravel(), 256)
        plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], 'gray')
        plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])
    plt.show()

if __name__ == '__main__':
    print("... Running Thresholding Directly...")
    # threshold()
    # adaptive_threshold()
    otsu_threshold('../Samples/Renders/blender_potato1.jpg')