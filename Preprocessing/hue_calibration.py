import cv2 as cv
import numpy as np
import math
import Preprocess as pr
import json

cap = cv.VideoCapture(1)
# cap = cv.VideoCapture('./assets/cam.avi')

hue_palette = cv.imread('./Preprocessing/assets/hue_palette.png', 1)

video_speed = 10
frame_counter = 0
PAUSE = False
QUIT = False

def load_variables():
    f = open('./variables.json')
    return json.load(f)


DATA = load_variables()

def updateRange(key):
    global DATA
    # Update HUE limits
    range = min(abs(DATA['HUE_MAX'] - DATA['HUE_MIN']), 180 - abs(DATA['HUE_MAX'] - DATA['HUE_MIN']))
    if key == 'w':
        DATA['HUE_MIN'] = DATA['HUE_MIN'] + 1 if DATA['HUE_MIN'] < 179 else 0
        DATA['HUE_MAX'] = DATA['HUE_MAX'] + 1 if DATA['HUE_MAX'] < 179 else 0
    if key == 's':
        DATA['HUE_MIN'] = DATA['HUE_MIN'] - 1 if DATA['HUE_MIN'] > 0 else 179
        DATA['HUE_MAX'] = DATA['HUE_MAX'] - 1 if DATA['HUE_MAX'] > 0 else 179
    if key == 'a' and range > 4:
        DATA['HUE_MAX'] = DATA['HUE_MAX'] - 1 if DATA['HUE_MAX'] > 0 else 179
    if key == 'd' and range < 90:
        DATA['HUE_MAX'] = DATA['HUE_MAX'] + 1 if DATA['HUE_MAX'] < 179 else 0
    print(f"New HUE range: ({DATA['HUE_MIN']} , {DATA['HUE_MAX']})")

    # Draw updated pallete
    RADIUS = 175
    CENTER = (205, 200)
    ang_min = -DATA['HUE_MIN'] * 2
    ang_max = -DATA['HUE_MAX'] * 2
    pt_min = (
        CENTER[0] + int(RADIUS * math.cos(ang_min * math.pi / 180)),
        CENTER[1] + int(RADIUS * math.sin(ang_min * math.pi / 180))
    )
    pt_max = (
        CENTER[0] + int(RADIUS * math.cos(ang_max * math.pi / 180)),
        CENTER[1] + int(RADIUS * math.sin(ang_max * math.pi / 180))
    )
    palette = np.copy(hue_palette)
    cv.line(palette, CENTER, pt_min, (100, 100, 100), 2)
    cv.line(palette, CENTER, pt_max, (100, 100, 100), 2)
    cv.imshow('Palette', palette)


# Show initial windows
updateRange(None)
ret, frame = cap.read()

# while (cap.isOpened()):
while (1):
    # Read next frame
    ret, frame = cap.read()
    # Flip image (TEMP)
    frame = cv.flip(frame, -1)
    frame_counter += 1
    if frame_counter == cap.get(cv.CAP_PROP_FRAME_COUNT):
        frame_counter = 0
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        continue
    if not ret:
        QUIT = True
    # Pause Loop
    while (2):
        # Wait key pressed event
        key = cv.waitKey(video_speed)

        # TODO add key to store the DATA['HUE_MIN'] and DATA['HUE_MAX'] values into the json
        # TODO add key to calibrate kernel size for blur

        if key == ord('w'):
            updateRange('w')
        if key == ord('a'):
            updateRange('a')
        if key == ord('s'):
            updateRange('s')
        if key == ord('d'):
            updateRange('d')
        if key == 32:  # Space
            PAUSE = not PAUSE
        elif key == 27:  # ESC
            QUIT = not QUIT

        # region FRAME PROCESSING

        hue, img_th = pr.hsv_otsu_threshold(frame, DATA['BLUR_SIZE'], DATA['HUE_MIN'], DATA['HUE_MAX'])

        pr.contour_detection(frame, img_th, DATA['MIN_PX_CONTOUR'])

        # endregion

        # region FRAME DISPLAY

        cv.imshow('Original', frame)
        cv.imshow('Hue channel', cv.resize(
            hue, (hue.shape[1] // 2, hue.shape[0] // 2), cv.INTER_AREA))
        cv.imshow('Threshold', cv.resize(
            img_th, (img_th.shape[1] // 2, img_th.shape[0] // 2), cv.INTER_AREA))

        # endregion

        # region VIDEO CONTROL
        if not PAUSE or QUIT:  # Go to next frame
            break
        # endregion
    if QUIT:
        break
# End process
cap.release()
cv.destroyAllWindows()
