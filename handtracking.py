import cv2
import mediapipe as mp
import time # to check the frame rate

cap = cv2.VideoCapture(0)
# Open camera with index, open video if given file name

mpHands = mp.solutions.hands
hands = mpHands.Hands()

mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    # camera input
    success, img = cap.read()

    # take a picture
    # TODO(shunxian): how often are we calling it
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks: # hand landmarks are detected, next step is to connect the dots
        for each_hand in results.multi_hand_landmarks: # for each hand, draw the landmarks separately
            for i, land_mark in enumerate(each_hand.landmark):
                # TODO(shunxian): x,y,z coordinates what is the z coordinate for?
                pass
            mpDraw.draw_landmarks(img, each_hand, mpHands.HAND_CONNECTIONS)

    # frame rate calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # program output
    cv2.imshow("Image", img)
    cv2.waitKey(1)