import cv2
import mediapipe as mp

if __name__ == '__main__':
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True,
                        model_complexity=1,
                        smooth_landmarks=True,
                        enable_segmentation=True,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
    drawing = mp.solutions.drawing_utils

    # read img BGR to RGB
    img = cv2.imread("1.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow("input", img)

    results = pose.process(img)
    drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow("keypoint", img)

    drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.waitKey(0)
    cv2.destroyAllWindows()