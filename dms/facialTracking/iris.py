import cv2
import time
import utils.conf as conf

from faceMesh import FaceMesh
from utils.utils import draw_iris


class Iris:
    """
    The object of iris, computing its position in the frame.

    Args:
        frame (numpy,ndarray): the input frame
        face_landmarks (mediapipe face landmarks object): contains the face landmarks coordinates
        id (list of int): the indices of eye in the landmarks
    """

    def __init__(self, frame, face_landmarks, id):

        self.frame = frame
        self.face_landmarks = face_landmarks.landmark
        self.id = id
        
        self.pos  = self._get_iris_pos()
    
    def _get_iris_pos(self):
        """Get the positions of iris."""
        h, w = self.frame.shape[:2]
        iris_pos = list()
        for id in self.id[-5:]:
            pos = self.face_landmarks[id]
            cx = int(pos.x * w)
            cy = int(pos.y * h)
            iris_pos.append([cx, cy])

        return iris_pos

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
        
        fm.process_frame(frame)
        if fm.mesh_result.multi_face_landmarks:
            for face_landmarks in fm.mesh_result.multi_face_landmarks:
                leftIris  = Iris(frame, face_landmarks, conf.LEFT_EYE)
                rightIris = Iris(frame, face_landmarks, conf.RIGHT_EYE)
                draw_iris(frame, leftIris.pos)
                draw_iris(frame, rightIris.pos)

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f'FPS: {int(fps)}', (30,30), 0, 0.8,
                    conf.TEXT_COLOR, 2, lineType=cv2.LINE_AA)

        cv2.imshow('Iris tracking', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
