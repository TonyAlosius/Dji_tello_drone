from djitellopy import tello
import KeyPressModule as kp
import numpy
import cv2
import time

me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamon()
me.takeoff()
me.send_rc_control(0, 0, 50, 0)
time.sleep(2.2)

w, h = 360, 240
# Forward and Backward Range
fbRange = [6200, 6800]
# proportional, integral, derivative
pid = [0.4, 0.4, 0]
preError = 0


def findFace(image):
    # https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
    # Load the Cascade Classifier
    faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
    # Convert the Image into Gray Scale
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect the face which returns the values (x, y, w, h)
    faces = faceCascade.detectMultiScale(imageGray, 1.2, 8)

    # Store the Co - Ordinates of the Image
    myFaceListCenter = []
    myFaceListArea = []
    for (x, y, w, h) in faces:
        # cv2.rectangle(image, start_point, end_point, color, thickness)
        # Bounding Box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        centerX = x + w // 2
        centerY = y + h // 2
        area = w * h
        # cv2.circle(image, start_point, end_point, color, thickness)
        # Center Dot
        cv2.circle(image, (centerX, centerY), 5, (0, 255, 0), cv2.FILLED)
        myFaceListCenter.append((centerX, centerY))
        myFaceListArea.append(area)
    if len(myFaceListArea) != 0:
        i = myFaceListArea.index(max(myFaceListArea))
        return image, [myFaceListCenter[i], myFaceListArea[i]]
    else:
        return image, [[0, 0], 0]


def trackFace(info, w, pid, preError):
    area = info[1]
    x, y = info[0]
    error = x - w // 2
    fb = 0
    speed = pid[0] * error + pid[1] * (error - preError)
    speed = int(numpy.clip(speed, -100, 100))
    if area > fbRange[0] and area < fbRange[1]:
        fb = 0
    # Range of area 6200 - 6800
    # Drone Too Close
    # When Drone comes closer area Increases
    # so move back the drone
    elif area > fbRange[1]:
        fb = -20
    # When Drone is far away area Decreases
    # As the size of the image decreases
    # so drone has to move forward
    # area == 0 that is no face detected stay stable
    # area != 0 move accordingly
    elif area < fbRange[0] and area != 0:
        fb = 20

    if x == 0:
        speed = 0
        error = 0
    # print(speed, fb)

    # me.send_rc_control(0, fb, 0, speed)
    return error


def getKeyboardInput():
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50
    if kp.getKey("LEFT"):
        lr = -speed
    elif kp.getKey("RIGHT"):
        lr = speed
    if kp.getKey("UP"):
        fb = speed
    elif kp.getKey("DOWN"):
        fb = -speed
    if kp.getKey("w"):
        ud = speed
    elif kp.getKey("s"):
        ud = -speed
    if kp.getKey("a"):
        yv = -speed
    elif kp.getKey("d"):
        yv = speed
    if kp.getKey("q"):
        me.land()
        sleep(3)
    if kp.getKey("e"):
        me.takeoff()

    return [lr, fb, ud, yv]


# capture = cv2.VideoCapture(0)
while True:
    vals = getKeyboardInput()
    me.send_rc_control(vals[0], vals[1], vals[2], vals[3])
    # _, image = capture.read()
    image = me.get_frame_read().frame
    image = cv2.resize(image, (w, h))
    image, info = findFace(image)
    preError = trackFace(info, w, pid, preError)
    # print('Center: ', center, 'Area: ', area)
    # print(type(info[1]))
    cv2.imshow("Output", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        me.land()
        break





