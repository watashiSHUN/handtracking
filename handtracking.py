import cv2
import mediapipe as mp
import time # to check the frame rate

class HandDetector:
    def __init__(self):
        self.mpHands = mp.solutions.hands
        # TODO(shunxian): repeatedly call method in a hand, remember the context?
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        if results.multi_hand_landmarks:
            for each_hand in results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, each_hand, self.mpHands.HAND_CONNECTIONS)
        # img is changed in place


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