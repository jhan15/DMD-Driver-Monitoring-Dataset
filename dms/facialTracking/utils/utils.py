import cv2
import utils.conf as conf


def draw_iris(frame, iris_pos, only_center=True):
    cv2.circle(frame, iris_pos[0], 2, conf.IRIS_COLOR, -1, lineType=cv2.LINE_AA)
    
    if not only_center:
        for pos in iris_pos[1:]:
            cv2.circle(frame, pos, 1, conf.IRIS_COLOR, -1, lineType=cv2.LINE_AA)

def draw_eye(frame, eye_pos):
    for pos in eye_pos:
        cv2.circle(frame, pos, 1, conf.EYE_COLOR, -1, lineType=cv2.LINE_AA)
