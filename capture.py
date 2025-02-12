import cv2

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    cv2.imshow('take photo',frame) #display the captured image
    key = cv2.waitKey(15) & 0xFF

    if key != 255:
        if key == 13:
            cv2.imwrite('images/cap.png', frame)
        cv2.destroyAllWindows()
        break

cap.release()