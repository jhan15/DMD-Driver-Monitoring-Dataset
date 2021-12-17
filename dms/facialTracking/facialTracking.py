import cv2
import time
import utils.conf as conf

from faceMesh import FaceMesh
from eye import Eye
from lips import Lips


class FacialTracker:

    def __init__(self):

        self.fm = FaceMesh()
    
    def process_frame(self, frame):
        self.gaze_status = None
        self.yawn_status = None

        self.fm.process_frame(frame)
        self.fm.draw_mesh_lips()

        if self.fm.mesh_result.multi_face_landmarks:
            for face_landmarks in self.fm.mesh_result.multi_face_landmarks:
                leftEye  = Eye(frame, face_landmarks, conf.LEFT_EYE)
                rightEye = Eye(frame, face_landmarks, conf.RIGHT_EYE)
                leftEye.iris.draw_iris(True)
                rightEye.iris.draw_iris(True)
                lips = Lips(frame, face_landmarks, conf.LIPS)

                if leftEye.eye_closed() or rightEye.eye_closed():
                    self.gaze_status = 'Eye closed'
                else:
                    if   leftEye.gaze_right()  and rightEye.gaze_right():
                        self.gaze_status = 'Gazing right'
                    elif leftEye.gaze_left()   and rightEye.gaze_left():
                        self.gaze_status = 'Gazing left'
                    elif leftEye.gaze_center() and rightEye.gaze_center():
                        self.gaze_status = 'Gazing center'
                
                if lips.mouth_open():
                    self.yawn_status = 'Yawning'

def main():
    cap = cv2.VideoCapture(conf.CAM_ID)
    cap.set(3, conf.FRAME_W)
    cap.set(4, conf.FRAME_H)
    facial_tracker = FacialTracker()
    ptime = 0
    ctime = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        facial_tracker.process_frame(frame)

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f'FPS: {int(fps)}', (30,30), 0, 0.6,
                    conf.TEXT_COLOR, 1, lineType=cv2.LINE_AA)
        
        if facial_tracker.gaze_status:
            cv2.putText(frame, f'{facial_tracker.gaze_status}', (30,70), 0, 0.8,
                        conf.WARN_COLOR, 2, lineType=cv2.LINE_AA)
        if facial_tracker.yawn_status:
            cv2.putText(frame, f'{facial_tracker.yawn_status}', (30,110), 0, 0.8,
                        conf.WARN_COLOR, 2, lineType=cv2.LINE_AA)

        cv2.imshow('Facial tracking', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
