# openCV image processing
import time

import cv2

# pose estimation
import mediapipe as mp
import numpy as np
import utils

output_window_name = "output"
# read video
cap = cv2.VideoCapture("/home/watashishun/dwhelper/20250114_205801.mp4")
# create a resizable window
cv2.namedWindow(output_window_name, cv2.WINDOW_NORMAL)

# Model
mpPose = mp.solutions.pose
pose = mpPose.Pose()

mpDraw = mp.solutions.drawing_utils

cTime = 0
pTime = 0

# TODO(shunxian): Add pause button (right click... on the video)
# TODO(shunxian): Make the following a module, and put `while True` in the main function
while True:
    success, img = cap.read()
    image_height, image_width, _ = img.shape
    # convert BGR to RGB
    # TODO(shunxian): what is BGR?
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose.process(imgRGB)

    if result.pose_landmarks:
        # landmarks are the points in the body
        # POSE_CONNECTIONS are the lines connecting the points
        # mpDraw.draw_landmarks(img, result.pose_landmarks, mpPose.POSE_CONNECTIONS)

        # Highlight right arm,
        # Right wrist,elbow,shoulder {x[0-1, ratio of the image, not pixel],y,z,visibility}
        landmark_indices = [
            mpPose.PoseLandmark.RIGHT_SHOULDER,
            mpPose.PoseLandmark.RIGHT_ELBOW,
            mpPose.PoseLandmark.RIGHT_WRIST,
        ]

        previous = None

        for index in landmark_indices:
            highlight = result.pose_landmarks.landmark[index]
            highlight.x = highlight.x * image_width
            highlight.y = highlight.y * image_height
            cv2.circle(
                img, (int(highlight.x), int(highlight.y)), 5, (255, 0, 255), cv2.FILLED
            )
            if previous is not None:
                cv2.line(
                    img,
                    (int(previous.x), int(previous.y)),
                    (int(highlight.x), int(highlight.y)),
                    (255, 0, 255),
                    3,
                )
            previous = highlight

        points = list(
            map(
                lambda x: (
                    int(result.pose_landmarks.landmark[x].x),
                    int(result.pose_landmarks.landmark[x].y),
                ),
                landmark_indices,
            )
        )
        angle = utils.computeAngle(*points) % 250
        percentage = np.interp(angle, (0, 180), (0, 100))
        # show the percentage
        bar = np.interp(angle, (0, 180), (400, 150))
        cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 255), 3)
        cv2.rectangle(img, (50, int(bar)), (85, 400), (0, 255, 255), cv2.FILLED)

        cv2.putText(
            img,
            str(int(angle)),
            (points[1][0] - 50, points[1][1] + 50),
            cv2.FONT_HERSHEY_PLAIN,
            3,
            (255, 0, 255),
            3,
        )

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # display frame rate
    cv2.putText(
        img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3
    )

    cv2.imshow(output_window_name, img)
    cv2.waitKey(1)  # 1 ms delay
