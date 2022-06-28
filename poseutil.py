import cv2
import mediapipe as mp
import math


class PoseDetector():
    '''
    人体姿势检测类
    '''
    def __init__(self,
                 static_image_mode=False,
                 upper_body_only=False,
                 smooth_landmarks=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        '''
        初始化
        :param static_image_mode: 是否是静态图片，默认为否
        :param upper_body_only: 是否是上半身，默认为否
        :param smooth_landmarks: 设置为True减少抖动
        :param min_detection_confidence:人员检测模型的最小置信度值，默认为0.5
        :param min_tracking_confidence:姿势可信标记的最小置信度值，默认为0.5
        '''
        self.static_image_mode = static_image_mode
        self.upper_body_only = upper_body_only
        self.smooth_landmarks = smooth_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        # 创建一个Pose对象用于检测人体姿势
        self.pose = mp.solutions.pose.Pose(self.static_image_mode, self.upper_body_only, self.smooth_landmarks,
                                           self.min_detection_confidence, self.min_tracking_confidence)
    def find_pose(self, img):
        '''
        检测姿势方法
        :param img: 一帧图像
        :param draw: 是否画出人体姿势节点和连接图
        :return: 处理过的图像
        '''
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # pose.process(imgRGB) 会识别这帧图片中的人体姿势数据，保存到self.results中
        self.results = self.pose.process(imgRGB)
        return img
    def draw_pose(self, img):
        mp.solutions.drawing_utils.draw_landmarks(img, self.results.pose_