import cv2 as cv
import numpy as np
import math
import Preprocess as pr
import json


def load_variables():
    f = open('./variables.json')
    return json.load(f)


DATA = load_variables()

cap = cv.VideoCapture(1)
frame_counter = 0

while (1):
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


        res_potato  = cv.resize(potato, (DATA['IMG_SIZE'], DATA['IMG_SIZE']))

        

        cv.imshow('Original Potato', potato)
        cv.imshow('Resized Potato', res_potato)
        print(res_potato.shape)


        for bb in boxes:
            (x, y, w, h) = bb
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)

    cv.imshow('Original', frame)
    cv.imshow('Threshold', cv.resize(
        img_th, (img_th.shape[1] // 2, img_th.shape[0] // 2), cv.INTER_AREA))

    if cv.waitKey(1) == 27:
        break

#     if frame_counter == cap.get(cv.CAP_PROP_FRAME_COUNT):
#         frame_counter = 0
#         cap.set(cv.CAP_PROP_POS_FRAMES, 0)
#         continue
#     if not ret:
#         QUIT = True
#     # Pause Loop
    # while (2):
    #     # Wait key pressed event
    #     key = cv.waitKey(video_speed)

#

#         # region FRAME DISPLAY

#         cv.imshow('Original', frame)
#         cv.imshow('Hue channel', cv.resize(
#             hue, (hue.shape[1] // 2, hue.shape[0] // 2), cv.INTER_AREA))
#         cv.imshow('Threshold', cv.resize(
#             img_th, (img_th.shape[1] // 2, img_th.shape[0] // 2), cv.INTER_AREA))

#         # endregion

#         # region VIDEO CONTROL
#         if not PAUSE or QUIT:  # Go to next frame
#             break
#         # endregion
#     if QUIT: break

# End process
cap.release()
cv.destroyAllWindows()
