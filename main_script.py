from enum import Enum

import cv2
import numpy as np

from vectoriser import methods, Shape
from gcode_viewer import view_gcode
from bot_interface import print_gcode

FRAME_TIME = 15
FPS = 500 // FRAME_TIME

filename = "latest_image.gcode"

circle_mask = cv2.imread("image_data/circle_mask.png") / 255.0
circle_mask_with_text = cv2.imread("image_data/circle_mask_with_text.png") / 255.0
countdown_1_mask = cv2.imread("image_data/countdown_1.png") / 255.0
countdown_2_mask = cv2.imread("image_data/countdown_2.png") / 255.0
countdown_3_mask = cv2.imread("image_data/countdown_3.png") / 255.0

shape_choice = cv2.imread("image_data/shape_choice.png")
processing_message = cv2.imread("image_data/processing.png")
drawing_message = cv2.imread("image_data/drawing.png")

ok_cancel_overlay = {
    Shape.CIRCLE: cv2.imread("image_data/ok_cancel_overlay_circle.png", cv2.IMREAD_UNCHANGED),
    Shape.SQUARE: cv2.imread("image_data/ok_cancel_overlay_square.png", cv2.IMREAD_UNCHANGED),
    Shape.SNEK: cv2.imread("image_data/ok_cancel_overlay_snek.png", cv2.IMREAD_UNCHANGED),
    Shape.STAR: cv2.imread("image_data/ok_cancel_overlay_star.png", cv2.IMREAD_UNCHANGED)
}

wait_overlay = cv2.imread("image_data/wait_overlay.png", cv2.IMREAD_UNCHANGED)
locked_overlay = cv2.imread("image_data/locked_overlay.png", cv2.IMREAD_UNCHANGED)

white = np.zeros((480, 640, 3), dtype=np.uint8) + 255

edge = np.zeros((480, 107, 3), dtype=np.uint8)



cap = cv2.VideoCapture(0)

window_name = 'Picassobot'

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

current_picture = None

class Mode(Enum):
    WAIT = "wait"
    LOCKED = "locked"
    CAPTURE = "capture"
    COUNTDOWN_1 = "countdown_1"
    COUNTDOWN_2 = "countdown_2"
    COUNTDOWN_3 = "countdown_3"
    FLASH = "flash"
    USE_PHOTO_QUESTION = "question"
    USE_PHOTO_CHOOSE = "choose"
    SHAPE_QUESTION = "shape_question"
    SHAPE_CHOOSE = "shape_choose"
    PROCESS = "process"
    DRAW = "draw"

shape = None
mouse_xy = (0,0)
mouse_click = False

def mouse_callback(event, x, y, flags, param):
    global mouse_xy, mouse_click
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_xy = (x, y)
        mouse_click = True

cv2.setMouseCallback(window_name, mouse_callback)

mode = Mode.LOCKED
photo = None


def pad_display(data):
    padded = np.concatenate((edge, data, edge), axis=1)
    cv2.imshow(window_name, padded)


def overlay_image(source, overlay):
    alpha = overlay[:,:,3:4] / 255.0

    return np.array(source*(1-alpha) + overlay[:,:,:3]*alpha, dtype=np.uint8)


counter = None

while True:

    ret, photo = cap.read()

    if mode == Mode.WAIT or mode == Mode.LOCKED:

        if ret:

            gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
            gray //= 3
            gray += (2 * 255) // 3

            photo = cv2.merge((gray, gray, gray))

        if mode == Mode.WAIT:
            next_mode = Mode.SHAPE_QUESTION
            pad_display(overlay_image(photo, wait_overlay))

        else:
            next_mode = Mode.LOCKED
            pad_display(overlay_image(photo, locked_overlay))

    elif mode == Mode.SHAPE_QUESTION:
        next_mode = Mode.SHAPE_CHOOSE

        pad_display(shape_choice)

    elif mode == Mode.SHAPE_CHOOSE:
        next_mode = None

        if mouse_xy[0] < 320:
            if mouse_xy[1] < 270:
                shape = Shape.CIRCLE
            else:
                shape = Shape.SNEK
        else:


            if mouse_xy[1] < 270:
                shape = Shape.SQUARE
            else:
                shape = Shape.STAR

        mode = Mode.CAPTURE



    elif mode == Mode.CAPTURE:

        next_mode = Mode.COUNTDOWN_3

        if ret:
            photo_with_circle = np.array(photo * circle_mask_with_text, dtype=np.uint8)
            pad_display(photo_with_circle)


    elif mode == Mode.COUNTDOWN_3:
        next_mode = None

        if counter is None:
            counter = FPS

        else:
            if counter < 1:
                mode = Mode.COUNTDOWN_2
                counter = None
                continue

        counter -= 1

        if ret:
            photo_with_circle = np.array(photo * countdown_3_mask, dtype=np.uint8)
            pad_display(photo_with_circle)


    elif mode == Mode.COUNTDOWN_2:
        next_mode = None

        if counter is None:
            counter = FPS

        else:
            if counter < 1:
                mode = Mode.COUNTDOWN_1
                counter = None
                continue

        counter -= 1

        if ret:
            photo_with_circle = np.array(photo * countdown_2_mask, dtype=np.uint8)
            pad_display(photo_with_circle)


    elif mode == Mode.COUNTDOWN_1:
        next_mode = None

        if counter is None:
            counter = FPS

        else:
            if counter < 1:
                mode = Mode.FLASH
                counter = None
                continue

        counter -= 1

        if ret:
            photo_with_circle = np.array(photo * countdown_1_mask, dtype=np.uint8)
            pad_display(photo_with_circle)


    elif mode == Mode.FLASH:
        next_mode = None

        if counter is None:
            counter = 2

        else:
            if counter < 1:
                mode = Mode.USE_PHOTO_QUESTION
                counter = None
                current_picture = photo
                continue

        counter -= 1

        pad_display(white)

    elif mode == Mode.USE_PHOTO_QUESTION:
        next_mode = Mode.USE_PHOTO_CHOOSE

        circled = np.array(current_picture * circle_mask, dtype=np.uint8)

        pad_display(overlay_image(circled, ok_cancel_overlay[shape]))

    elif mode == Mode.USE_PHOTO_CHOOSE:
        next_mode = None

        if mouse_xy[0] < 320:
            mode = Mode.WAIT
        else:
            mode = Mode.PROCESS

        pad_display(drawing_message)


    elif mode == Mode.PROCESS:
        next_mode = None

        converter = methods[shape]()

        with open(filename, 'w') as file:
            file.write(converter.gcode(current_picture, show=None))
            file.write("\n")

        view_gcode(filename, output_filename="preview.png")

        pad_display(drawing_message)

        mode = Mode.DRAW

    elif mode == Mode.DRAW:

        print_gcode(filename)

        mode = Mode.WAIT

    key = cv2.waitKey(15) & 0xFF

    if key == 255:
        if mouse_click:

            if next_mode is not None:
                mode = next_mode

            mouse_click = False

    elif key == 13:
        mode = next_mode
        mouse_click = False

    elif key == ord("l"):
        if mode == Mode.LOCKED:
            mode = Mode.WAIT
        else:
            mode = Mode.LOCKED

    elif key == ord('q'):
        break



cv2.destroyAllWindows()
cap.release()