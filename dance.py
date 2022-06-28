import cv2
import time
import numpy as np
from poseutil import PoseDetector

# 打开一个视频文件
cap = cv2.VideoCapture('videos/dance5.mp4')
# 姿势识别器
d = PoseDetector()
# 视频宽度高度和fps
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
# 录制视频设置
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('videos/dance5_output.mp4', fourcc, fps, (width, height))
begin = time.time()
while True:
    success, img = cap.read()
    if success:
        # 人体姿势识别
        d.find_pose(img)
        # 每隔10秒将视频变黑
        now = time.time()
        t = int(now - begin)
        if t // 10 % 2 == 1:
            mask = np.zeros(img.shape[:2], dtype="uint8")
            img = cv2.bitwise_and(img, img, mask=mask)
        # 画人体姿势识别图
        d.draw_pose(img)
        cv2.imshow('Dance', img)
        # 保存视频
        out.write(img)
    else:
        break
    # 按q键退出
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
out.release()
cap.release()
cv2.destroyAllWindows()