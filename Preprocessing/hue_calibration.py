import cv2 as cv
import numpy as np
import math
import Preprocessing.Thresholding as pr_th

# cap = cv.VideoCapture(1)
cap = cv.VideoCapture('./assets/cam.avi')

hue_palette = cv.imread('./assets/hue_palette.png', 1)

video_speed = 15
frame_counter = 0
PAUSE = False
QUIT = False
HUE_MIN = 0
HUE_MAX = 20



def updateRange(key):
    global HUE_MIN, HUE_MAX
    # Update HUE limits
    range = min(abs(HUE_MAX - HUE_MIN), 180 - abs(HUE_MAX - HUE_MIN))
    if key == 'w':
        HUE_MIN = HUE_MIN + 1 if HUE_MIN < 179 else 0
        HUE_MAX = HUE_MAX + 1 if HUE_MAX < 179 else 0
    if key == 's':
        HUE_MIN = HUE_MIN - 1 if HUE_MIN > 0 else 179
        HUE_MAX = HUE_MAX - 1 if HUE_MAX > 0 else 179
    if key == 'a' and range > 4:
        HUE_MAX = HUE_MAX - 1 if HUE_MAX > 0 else 179
    if key == 'd' and range < 60:
        HUE_MAX = HUE_MAX + 1 if HUE_MAX < 179 else 0
    (f"New HUE range: ({HUE_MIN} , {HUE_MAX})")
    # Draw updated pallete
    RADIUS = 175
    CENTER = (205, 200)
    ang_min = -HUE_MIN * 2
    ang_max = -HUE_MAX * 2
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


def threshold(img, hue_min, hue_max):
    th_min = (img >= hue_min)
    th_max = (img <= hue_max)
    th = (th_min * th_max) if hue_min < hue_max else (th_min + th_max)
    th = th * 255
    return th.astype(np.uint8)


# Show initial windows
updateRange(None)
ret, frame = cap.read()

# while (cap.isOpened()):
while (1):
    # Read next frame
    ret, frame = cap.read()
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

        # img_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # hue = img_hsv[:, :, 0]
        # 
        # img_th = threshold(hue, HUE_MIN, HUE_MAX)

        hue, img_th = pr_th.hsv_otsu_threshold(frame)

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
    if QUIT: break
# End process
cap.release()
cv.destroyAllWindows()
