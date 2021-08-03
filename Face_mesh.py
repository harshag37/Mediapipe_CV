import mediapipe as mp
import cv2
import time
class face_mesh():
    def __init__(self):
        self.mpFacedMesh=mp.solutions.face_mesh
        self.FaceMesh=self.mpFacedMesh.FaceMesh()
        self.mpdraw=mp.solutions.drawing_utils
        self.draw_specs=self.mpdraw.DrawingSpec(thickness=1,circle_radius=2)
    def detect_face(self,img,draw=True):
        imgrgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.result=self.FaceMesh.process(imgrgb)
        lms=[]
        if draw:
            if self.result.multi_face_landmarks:
                for facelms in self.result.multi_face_landmarks:
                    self.mpdraw.draw_landmarks(img,facelms,self.mpFacedMesh.FACE_CONNECTIONS
                    ,self.draw_specs,self.draw_specs)
                    lms.append(facelms)
        return img,lms
def main():
    ptime=0
    ctime=0
    vid=cv2.VideoCapture(0)
    detector=face_mesh()
    while True:
        check,img=vid.read()
        img,lms=detector.detect_face(img)
        ctime=time.time()
        fps=1/(ctime-ptime)
        ptime=ctime
        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,0))
        cv2.imshow("image",img)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
if __name__=="__main__":
    main()
