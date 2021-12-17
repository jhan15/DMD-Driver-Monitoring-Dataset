import cv2
import time
import utils.conf as conf

from faceMesh import FaceMesh
from iris import Iris
from utils.utils import draw_iris, draw_eye


class Eye:
    """
    The object of eye, computing its features from face landmarks.

    Args:
        frame (numpy,ndarray): the input frame
        face_landmarks (mediapipe face landmarks object): contains the face landmarks coordinates
        id (list of int): the indices of eye in the landmarks
    """

    def __init__(self, frame, face_landmarks, id):

        self.frame = frame
        self.face_landmarks = face_landmarks.landmark
        self.id = id

        self.iris = Iris(frame, face_landmarks, id)
        self.pos  = self._get_eye_pos()
        self.iris_relative_to_eye = self._get_ratio()
    
    def _get_eye_pos(self):
        """Get the left, right, top, and bottom positions of eye."""
        h, w = self.frame.shape[:2]
        eye_pos = list()
        for id in self.id[:4]:
            pos = self.face_landmarks[id]
            cx = int(pos.x * w)
            cy = int(pos.y * h)
            eye_pos.append([cx, cy])

        return eye_pos
    
    def _get_ratio(self):
        """Get the ratio of iris relative to eye."""
        ratiol = (self.pos[0][0] - self.iris.pos[1][0]) / (self.pos[0][0] - self.pos[1][0])
        ratioc = (self.pos[0][0] - self.iris.pos[0][0]) / (self.pos[0][0] - self.pos[1][0])
        ratior = (self.pos[0][0] - self.iris.pos[3][0]) / (self.pos[0][0] - self.pos[1][0])

        return [ratiol, ratioc, ratior]
    
    def gaze_left(self, threshold=conf.GAZE_LEFT):
        """Check whether gazing left."""
        return self.iris_relative_to_eye[0] < threshold
    
    def gaze_right(self, threshold=conf.GAZE_RIGHT):
        """Check whether gaze right."""
        return self.iris_relative_to_eye[2] > threshold
    
    def gaze_center(self):
        """Check whether gazing center."""
        return not self.gaze_left() and not self.gaze_right()

def main():
    cap = cv2.VideoCapture(conf.CAM_ID)
    cap.set(3, conf.FRAME_W)
    cap.set(4, conf.FRAME_H)
    fm = FaceMesh()
    ptime = 0
    ctime = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        text = 'unkown'
        
        fm.process_frame(frame)
        if fm.mesh_result.multi_face_landmarks:
            for face_landmarks in fm.mesh_result.multi_face_landmarks:
                leftEye  = Eye(frame, face_landmarks, conf.LEFT_EYE)
                rightEye = Eye(frame, face_landmarks, conf.RIGHT_EYE)
                draw_iris(frame, leftEye.iris.pos)
                draw_iris(frame, rightEye.iris.pos)

                if leftEye.gaze_left() and rightEye.gaze_left():
                    text = 'left'
                elif leftEye.gaze_right() and rightEye.gaze_right():
                    text = 'right'
                elif leftEye.gaze_center() and rightEye.gaze_center():
                    text = 'center'

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f'FPS: {int(fps)}', (30,30), 0, 0.8,
                    conf.TEXT_COLOR, 2, lineType=cv2.LINE_AA)
        cv2.putText(frame, f'Gazing {text}', (30,80), 0, 0.8,
                    conf.TEXT_COLOR, 2, lineType=cv2.LINE_AA)

        cv2.imshow('Face Mesh', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
