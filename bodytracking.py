import cv2
import mediapipe as mp
import time

output_window_name = "output"
# read video
cap = cv2.VideoCapture("/home/watashishun/Documents/haoran_climbing.mp4")
# create a resizable window
cv2.namedWindow(output_window_name, cv2.WINDOW_NORMAL)

mpPose = mp.solutions.pose
pose = mpPose.Pose()

mpDraw = mp.solutions.drawing_utils

cTime = 0
pTime = 0

while True:
    success, img = cap.read()
    # convert BGR to RGB 
    # TODO(shunxian): what is BGR?
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose.process(imgRGB)

    if (result.pose_landmarks):
        mpDraw.draw_landmarks(img, result.pose_landmarks, mpPose.POSE_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    #display frame rate
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow(output_window_name, img)
    cv2.waitKey(1) # 1 ms delay