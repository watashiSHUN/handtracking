import cv2
import mediapipe as mp
import time # to check the frame rate
from enum import Enum

class FingerType(Enum):
    THUMB = 0
    INDEX = 1
    MIDDLE = 2
    RING = 3
    PINKY = 4
        
class Finger:
    def __init__(self, landmarks, finger_type):
        self.finger_type = finger_type
        self.tip = landmarks[3]
        self.dip = landmarks[2]
        self.pip = landmarks[1]
        self.mcp = landmarks[0]

class Hand:
    def __init__(self, hand_landmarks):
        # landmark index to finger mapping
        # NOTE, when landmark is outside of the picture, the value is also outside
        # has landmark, but the value is None
        # NOTE, use `mp_hands.HandLandmark.INDEX_FINGER_TIP``
        self.thumb = Finger(hand_landmarks[1:5], FingerType.THUMB)
        self.index = Finger(hand_landmarks[5:9], FingerType.INDEX)
        self.middle = Finger(hand_landmarks[9:13], FingerType.MIDDLE)
        self.ring = Finger(hand_landmarks[13:17], FingerType.RING)
        self.pinky = Finger(hand_landmarks[17:21], FingerType.PINKY)

class HandDetector:
    def __init__(self):
        self.mpHands = mp.solutions.hands
        # TODO(shunxian): repeatedly call method in a hand, remember the context?
        self.hands = self.mpHands.Hands(min_detection_confidence=0.75)
        self.mpDraw = mp.solutions.drawing_utils

    # return the hand positions
    def findHands(self, img, draw=True):
        return_value = []
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        if results.multi_hand_landmarks:
            for each_hand in results.multi_hand_landmarks:
                return_value.append(Hand(each_hand.landmark))
                if draw:
                    # img is changed in place
                    self.mpDraw.draw_landmarks(img, each_hand, self.mpHands.HAND_CONNECTIONS)
        # TODO(shunxian): left hand or right hand
        return return_value


# driver code
def main():
    cap = cv2.VideoCapture(0)
    hand_detector = HandDetector()
    # Open camera with index, open video if given file name
    
    pTime = 0
    cTime = 0
    while True:
        # camera input
        success, img = cap.read()

        hand_detector.findHands(img)
        
        # frame rate calculation
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        # program output
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()