import cv2 as cv
import Preprocess as pr

import tensorflow as tf
# import tensorflow.keras as keras

import numpy as np
import json


def load_variables():
    f = open('./variables.json')
    return json.load(f)


def load_model():
    model = tf.keras.models.Sequential([
        # Conv layers
        tf.keras.layers.Conv2D(
            16, (3, 3), activation='relu', input_shape=(250, 250, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Flatten layer
        tf.keras.layers.Flatten(),
        # Fully connected layers
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        # Output neuron
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.load_weights('./CNN/checkpoints/last_training')

    return model


DATA = load_variables()
MODEL = load_model()

# Select video source
# cap = cv.VideoCapture(0)
cap = cv.VideoCapture('./Preprocessing/assets/cam.avi')


def setImageResolution():
    # Default res is 640x480 px, but the camera supports a resolution of 1280x720 px
    # Use next function to update image resolution
    w = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    print("Frame default resolution:", w, h)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 2000)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 2000)
    w = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    print("NEW Frame default resolution:", w, h)


video_speed = 10
frame_counter = 0
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
        img_th, DATA['MIN_PX_CONTOUR'], DATA['DOWN_BOUND'], DATA['UP_BOUND'], DATA['BOUNDING_BOX_MARGIN'])

    if len(boxes):

        # Extract first image
        (x, y, w, h) = boxes[0]
        potato = frame[y:y+h, x:x+w, :]

        res_potato = cv.resize(potato, (DATA['IMG_SIZE'], DATA['IMG_SIZE']))

        pred = tf.keras.preprocessing.image.img_to_array(res_potato) / 255
        pred = np.expand_dims(pred, axis=0)
        images = np.vstack([pred])

        classes = MODEL.predict(images, batch_size=1)
        label = 'Healthy' if classes[0, 0] > 0.5 else 'Damaged'
        color = (255, 0, 0) if classes[0, 0] > 0.5 else (0, 0, 255)
        # percent = classes[0,0] if classes[0,0] > 0.5 else 1 - classes[0,0]

        cv.putText(frame, label, (x, y-10),
                   cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv.putText(frame, str(classes[0, 0]), (x, y+h+20),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)
        cv.rectangle(frame, (x, y), (x+w, y+h), color, 3)

        for bb in boxes[1:]:
            (x, y, w, h) = bb
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), 3)

        cv.imshow('Resized Potato', res_potato)

    cv.imshow('Original', frame)

    # Exit program
    if QUIT:
        break


# End process
cap.release()
cv.destroyAllWindows()
