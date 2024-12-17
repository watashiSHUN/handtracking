import cv2
import time
import numpy as np # NOTE: why import numpy?
import handtracking as ht


# draw a line between index finger and thumb
# use the length to determine the volume

# NOTE(shunxian): z coordinate, if I hold the hand and just tilt it, volume should not change
# distance is sqrt((x-x0)^2 + (y-y0)^2 + (z-z0)^2)

# NOTE(shunxian): x, y coordinates are pixels, but what about z?
# z is changing, the wrist is always at 0 z coordinate (https://mediapipe.readthedocs.io/en/latest/solutions/hands.html#multi-hand-landmarks)

# 4 is external camera
cap = cv2.VideoCapture(0) # 0 for webcam, 4 for external camera

pTime, cTime = 0, 0

hand_detector = ht.HandDetector()

while True:
    success, img = cap.read()

    # Draw hand in the image 
    hands = hand_detector.findHands(img)
    if hands:
        main_hand = hands[0]
        a = main_hand.index.tip
        b = main_hand.thumb.tip
        # draw a line between index finger and thumb
        h,w,c = img.shape
        if 0 <= a.x < 1 and 0 <= a.y < 1 and 0 <= b.x < 1 and 0 <= b.y < 1:
            cv2.line(img, (int(a.x*w), int(a.y*h)), (int(b.x*w), int(b.y*h)), (255, 0, 255), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime) # first display should be very wrong
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1) # 1 ms delay