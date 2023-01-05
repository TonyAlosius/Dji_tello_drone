import numpy as np
from djitellopy import tello
import cv2

me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamon()

# me.takeoff()

cap = cv2.VideoCapture(1)
hsvVals = [0, 0, 188, 179, 33, 245]
sensors = 3
threshold = 0.2
width, height = 480, 360
sensitivity = 3  # if number is high less sensitive
weights = [-25, -15, 0, 15, 25]
fSpeed = 15
curve = 0


def thresholding(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([hsvVals[0], hsvVals[1], hsvVals[2]])
    upper = np.array([hsvVals[3], hsvVals[4], hsvVals[5]])
    mask = cv2.inRange(hsv, lower, upper)
    return mask


def getContours(imgThreshold, image):
    cx = 0
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        biggest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(biggest)
        cx = x + w // 2
        cy = y + h // 2
        cv2.drawContours(image, biggest, -1, (255, 0, 255), 7)
        cv2.circle(image, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
    return cx


def getSensorOutput(imgThreshold, sensors):
    images = np.hsplit(imgThreshold, sensors)
    totalPixels = (image.shape[1] // sensors) * image.shape[0]
    senOut = []
    for x, im in enumerate(images):
        pixelCount = cv2.countNonZero(im)
        if pixelCount > threshold * totalPixels:
            senOut.append(1)
        else:
            senOut.append(0)
        # cv2.imshow(str(x), im)
    # print(senOut)
    return senOut


def sendCommands(senOut, cx):
    global curve
    # TRANSLATION
    lr = (cx - width // 2) // sensitivity
    lr = int(np.clip(lr, -10, 10))
    if 2 > lr > -2: lr = 0
    # Rotation
    if senOut == [1, 0, 0]: 
        curve = weights[0]
    elif senOut == [1, 1, 0]: 
        curve = weights[1]
    elif senOut == [0, 1, 0]: 
        curve = weights[2]
    elif senOut == [0, 1, 1]: 
        curve = weights[3]
    elif senOut == [0, 0, 1]: 
        curve = weights[4]
    elif senOut == [0, 0, 0]: 
        curve = weights[2]
    elif senOut == [1, 1, 1]: 
        curve = weights[2]
    elif senOut == [1, 0, 1]: 
        curve = weights[2]
    me.send_rc_control(lr, fSpeed, 0, curve)


while True:
    # _, image = cap.read()
    image = me.get_frame_read().frame
    image = cv2.resize(image, (width, height))
    image = cv2.flip(image, 0)
    imgThreshold = thresholding(image)
    cx = getContours(imgThreshold, image)  # For Translation
    senOut = getSensorOutput(imgThreshold, sensors)  # Rotation
    sendCommands(senOut, cx)
    cv2.imshow("Output", image)
    cv2.imshow("Path", imgThreshold)
    cv2.waitKey(1)