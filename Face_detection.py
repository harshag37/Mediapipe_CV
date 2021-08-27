import mediapipe as mp
import cv2
import time
class Face_detection():
    def __init__(self):
        self.mpFacedetection=mp.solutions.face_detection
        self.FaceDetection=self.mpFacedetection.FaceDetection()
        self.mpdraw=mp.solutions.drawing_utils
    def detect_face(self,img,draw=True):
        imgrgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.result=self.FaceDetection.process(imgrgb)
        if draw:
            if self.result.detections:
                for id,detection in enumerate(self.result.detections):
                    self.mpdraw.draw_detection(img,detection)
        return img
def main():
    ptime=0
    ctime=0
    vid=cv2.VideoCapture(0)
    detector=Face_detection()
    while True:
        check,img=vid.read()
        img=detector.detect_face(img)
        ctime=time.time()
        fps=1/(ctime-ptime)
        ptime=ctime
        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,0))
        cv2.imshow("image",img)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
if __name__=="__main__":
    main()