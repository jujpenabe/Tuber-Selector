import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def hsv_otsu_threshold(image, blur, value_min, value_max, channel=0):

    # Aplicar filtro paso bajo (blur)
    image = cv.blur(image, (blur, blur), 0)

    # Convertir imágen a espacio de color HSV
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    th_min = (hsv[:, :, channel] >= value_min)
    th_max = (hsv[:, :, channel] <= value_max)
    th = (th_min * th_max) if value_min < value_max else (th_min + th_max)
    th = th * 255  # TODO check if uint8 is really necessary
    th = th.astype(np.uint8)

    # Aplicar binarización OTSU
    _, thr = cv.threshold(th, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return hsv[:, :, channel], thr


def box_detection(img_th, min_pixels, margin = 0):
    boxes = []
    # Find all contours
    contours, hierarchy = cv.findContours(
        img_th, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # Filter contours
    for cont in contours:
        (x, y, w, h) = cv.boundingRect(cont)
        c1 = w > min_pixels and h > min_pixels
        c2 = x-margin >= 0 and x + w + margin <= img_th.shape[1]
        c3 = y-margin >= 0 and y + h + margin <= img_th.shape[0]
        if c1 and c2 and c3:
            boxes.append((x-margin, y-margin, w+2*margin, h+2*margin))
    return boxes


def contour_detection(source, img_th, min_pixels):
    # Find all contours
    contours, hierarchy = cv.findContours(
        img_th, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # Filter contours
    for cont in contours:
        (x, y, w, h) = cv.boundingRect(cont)
        if w > min_pixels and h > min_pixels:
            cv.drawContours(source, [cont],  0, (0, 0, 255), 3)
