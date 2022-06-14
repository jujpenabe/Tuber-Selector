import cv2 as cv
import numpy as np
import Preprocess as pr
import json
import time


def load_variables():
    f = open('../variables.json')
    return json.load(f)


DATA = load_variables()

# Select video source
cap = cv.VideoCapture(0)
# cap = cv.VideoCapture('./assets/cam.avi')

video_speed = 10
frame_counter = 0
aux_timestamp = 0
img_count = 332
STORE_IMG = False
QUIT = False

while (1):

    # Wait key pressed event
    key = cv.waitKey(video_speed)
    if key == 32:  # Space
        STORE_IMG = not STORE_IMG
        print('Sotre images', 'ENABLED' if STORE_IMG else 'DISABLED')
    elif key == 27:  # ESC
        QUIT = not QUIT

    # Read next frame
    ret, frame = cap.read()
    # Flip image
    frame = cv.flip(frame, -1)
    frame_counter += 1

    hue, img_th = pr.hsv_otsu_threshold(
        frame, DATA['BLUR_SIZE'], DATA['HUE_MIN'], DATA['HUE_MAX'])

    boxes = pr.box_detection(
        img_th, DATA['MIN_PX_CONTOUR'], DATA['BOUNDING_BOX_MARGIN'])

    if len(boxes):

        # Extract first image
        (x, y, w, h) = boxes[0]
        potato = frame[y:y+h, x:x+w, :]

        res_potato = cv.resize(potato, (DATA['IMG_SIZE'], DATA['IMG_SIZE']))

        cv.imshow('Original Potato', potato)
        cv.imshow('Resized Potato', res_potato)

        for bb in boxes:
            (x, y, w, h) = bb
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)

    cv.imshow('Original', frame)
    cv.imshow('Hue channel', cv.resize(
        hue, (hue.shape[1] // 2, hue.shape[0] // 2), cv.INTER_AREA))
    cv.imshow('Threshold', cv.resize(
        img_th, (img_th.shape[1] // 2, img_th.shape[0] // 2), cv.INTER_AREA))

    # TIME
    timestamp = time.perf_counter()
    if timestamp - aux_timestamp >= DATA['PHOTO_INTERVAL']:
        aux_timestamp = timestamp
        if STORE_IMG:
            filename = f'img_{img_count:04d}.jpg'
            print(f'New image Capture {filename} on {timestamp} sec.')
            cv.imwrite('images/' + filename, res_potato)
            img_count += 1

    # Exit program
    if QUIT:
        break


# End process
cap.release()
cv.destroyAllWindows()
