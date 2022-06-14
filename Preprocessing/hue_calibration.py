import cv2 as cv
import numpy as np
import math
import Preprocess as pr
import json

# Select video source
# cap = cv.VideoCapture(1)
cap = cv.VideoCapture('./assets/cam.avi')

hue_palette = cv.imread('./assets/hue_palette.png', 1)

video_speed = 10
frame_counter = 0
HIDE = False
FLIP = False
BLINK = False
PAUSE = False
QUIT = False
DATA = {}


def load_variables():
    f = open('../variables.json')
    return json.load(f)


DATA = load_variables()


def update_blink(counter, boolean, wait=30):
    if boolean:
        if counter % wait == 0:
            boolean = not boolean
    else:
        if counter % int(wait/3) == 0:
            boolean = not boolean
    return boolean


def update_data(key):
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
    if key == 'q':
        DATA['BLUR_SIZE'] = DATA['BLUR_SIZE'] - 1 if DATA['BLUR_SIZE'] > 1 else 1
    if key == 'e':
        DATA['BLUR_SIZE'] = DATA['BLUR_SIZE'] + 1 if DATA['BLUR_SIZE'] < 150 else 150
    if key == 'i':
        DATA['UP_BOUND'] = DATA['UP_BOUND'] - 1 if DATA['UP_BOUND'] > 0 else 0
        DATA['DOWN_BOUND'] = DATA['DOWN_BOUND'] - 1 if DATA['UP_BOUND'] > 0 else 0 
    if key == 'k':
        DATA['UP_BOUND'] = DATA['UP_BOUND'] + 1 if DATA['UP_BOUND'] < 480 else 480
        DATA['DOWN_BOUND'] = DATA['DOWN_BOUND'] + 1 if DATA['DOWN_BOUND'] < 480 else 480
    if key == 'j':
        DATA['UP_BOUND'] = DATA['UP_BOUND'] + 1 if DATA['UP_BOUND'] < 480 and DATA['UP_BOUND'] < DATA['DOWN_BOUND'] else DATA['UP_BOUND']
        DATA['DOWN_BOUND'] = DATA['DOWN_BOUND'] - 1 if DATA['DOWN_BOUND'] > 0 and DATA['DOWN_BOUND'] > DATA['UP_BOUND'] else DATA['DOWN_BOUND']
    if key == 'l':
        DATA['UP_BOUND'] = DATA['UP_BOUND'] - 1 if DATA['UP_BOUND'] > 0 else 0
        DATA['DOWN_BOUND'] = DATA['DOWN_BOUND'] + 1 if DATA['DOWN_BOUND'] < 480 else 480
    
    print(f"NEW HUE range: ({DATA['HUE_MIN']} , {DATA['HUE_MAX']})  NEW BLUR size: {DATA['BLUR_SIZE']}")

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


def save_data(data):
    out_file = open("../variables.json", "w")

    json.dump(data, out_file, indent=5)

    out_file.close()


# Show initial windows
update_data(None)
ret, frame = cap.read()

# while (cap.isOpened()):
while (1):
    # Read next frame
    ret, frame = cap.read()
    # Flip image (TEMP)
    if FLIP:
        frame = cv.flip(frame, -1)
    frame_counter += 1
    # Check if Video has reached the end
    if frame_counter == cap.get(cv.CAP_PROP_FRAME_COUNT):
        frame_counter = 0
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)  # Repeat Video
        continue
    if not ret:
        QUIT = True
    # Pause Loop
    while (1):
        # Wait key pressed event
        key = cv.waitKey(video_speed)
        if key == ord('w'):
            update_data('w')
        if key == ord('a'):
            update_data('a')
        if key == ord('s'):
            update_data('s')
        if key == ord('d'):
            update_data('d')
        if key == ord('q'):
            update_data('q')
        if key == ord('e'):
            update_data('e')
        if key == 13:  # Enter
            save_data(DATA)
        if key == ord('f'):
            FLIP = not FLIP
        if key == ord('h'):
            HIDE = not HIDE
        if key == ord('i'):
            update_data('i')
        if key == ord('j'):
            update_data('j')
        if key == ord('k'):
            update_data('k')
        if key == ord('l'):
            update_data('l')

        # TODO: Add Preview lines for centroid range
        # TODO: Add centroid range data (x1, x2, y1, y2) to json
        # region FRAME PROCESSING

        hue, img_th = pr.hsv_otsu_threshold(frame, DATA['BLUR_SIZE'], DATA['HUE_MIN'], DATA['HUE_MAX'])
        if not PAUSE:
            pr.contour_detection(frame, img_th, DATA['MIN_PX_CONTOUR'])

        # endregion

        # region FRAME DISPLAY
        current = np.copy(frame)
        # region DATA PREVIEW
        if not HIDE:
            f = '{0}: {1:>3}'
            BLINK = update_blink(frame_counter, BLINK, 45)
            cv.putText(current, "D A T A", (5, 22),
                       cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
            cv.putText(current, f.format("Blur Size", DATA['BLUR_SIZE']), (5, 40),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 0), 1)
            cv.putText(current, f.format("HUE MIN", DATA['HUE_MIN']), (5, 55),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (64, 128, 0), 1)
            cv.putText(current, f.format("HUE MAX", DATA['HUE_MAX']), (5, 70),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (64, 0, 128), 1)
            cv.putText(current, "Press ENTER to SAVE", (5, 85),
                       cv.FONT_HERSHEY_SIMPLEX, 0.4, (64, 64, 64), 1)
            # CALIBRATING
            if BLINK or PAUSE:
                cv.putText(current, "CALIBRATING", (int(current.shape[1] / 2 - (len("CALIBRATING") * 10)), 25),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # Controls panel
            cv.putText(current, "C O N T R O L S", (int(current.shape[1] - (len("CONTROLS") * 20) - 60), 22),
                       cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
            cv.putText(current, "[W]: +MIN +MAX", (int(current.shape[1] - (len("CONTROLLS") * 20) - 5), 40),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 0), 1)
            cv.putText(current, "HUE:  [S]: -MIN -MAX", (int(current.shape[1] - (len("CONTROLLS") * 20) - 58), 56),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 128), 1)
            cv.putText(current, "[A]: -MAX  [D]: +MAX", (int(current.shape[1] - (len("CONTROLLS") * 20) - 5), 72),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 64, 64), 1)
            cv.putText(current, "BLUR:  [Q]: -Blur  [E]: +Blur", (int(current.shape[1] - (len("CONTROLLS") * 20) - 67), 88),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 0), 1)
            cv.putText(current, "VIDEO:  [F]: Flip", (int(current.shape[1] - (len("CONTROLLS") * 20) - 70), 104),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv.putText(current, "[SPACE]: Pause", (int(current.shape[1] - (len("CONTROLLS") * 20) - 5), 120),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv.putText(current, "PROGRAM:  [H]: Hide Interface", (int(current.shape[1] - (len("CONTROLLS") * 25) - 55), 136),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv.putText(current, "[ENTER]: Save",  (int(current.shape[1] - (len("CONTROLLS") * 20) - 5),152),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 64, 64), 1)
            cv.putText(current, "[ESC]: EXIT",  (int(current.shape[1] - (len("CONTROLLS") * 20) - 5),168),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (64, 64, 255), 1)

            cv.line(current, (0, DATA['UP_BOUND']), (current.shape[1], DATA['UP_BOUND']), (255, 0, 0), 2)
            cv.line(current, (0, DATA['DOWN_BOUND']), (current.shape[1], DATA['DOWN_BOUND']), (255, 0, 0), 2)
# endregion
        cv.imshow('Original', current)
        cv.imshow('Hue channel', cv.resize(
            hue, (hue.shape[1] // 2, hue.shape[0] // 2), cv.INTER_AREA))
        cv.imshow('Threshold', cv.resize(
            img_th, (img_th.shape[1] // 2, img_th.shape[0] // 2), cv.INTER_AREA))

        # endregion

        # region VIDEO CONTROL
        if key == 32:  # Space
            PAUSE = not PAUSE
        if key == 27:  # ESC
            QUIT = not QUIT

        if not PAUSE or QUIT:  # Go to next frame
            break
        elif key == -1:
            continue
        else:
            print("You pressed: ", key)
        # endregion

    if QUIT:
        break
# End process
cap.release()
cv.destroyAllWindows()
