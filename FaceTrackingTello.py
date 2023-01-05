from utils import *
import cv2

w, h = 1290, 720
myDrone = initTello()
while True:

    # Step 1
    Image = telloGetFrame(myDrone, w, h)
    cv2.imshow("Live Streaming", Image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        myDrone.land()
        break