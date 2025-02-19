import cv2

cap = cv2.VideoCapture(1)

window_name = 'take photo'

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:

    ret, frame = cap.read()

    cv2.imshow(window_name,frame) #display the captured image
    key = cv2.waitKey(15) & 0xFF

    if key != 255:
        if key == 13:
            cv2.imwrite('images/cap.png', frame)
        cv2.destroyAllWindows()
        break

cap.release()