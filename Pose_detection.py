import mediapipe as mp
import cv2
import time
class pose_detection():
    def __init__(self):
        self.mpPOSE=mp.solutions.pose
        self.Pose=self.mpPOSE.Pose()
        self.mpdraw=mp.solutions.drawing_utils
    def Detect_Pose(self,img,draw=True):
        imgrgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.result=self.Pose.process(imgrgb)
        if draw:
            if self.result.pose_landmarks:
                self.mpdraw.draw_landmarks(img,self.result.pose_landmarks,self.mpPOSE.POSE_CONNECTIONS)
        return img
    def find_pos(self,img,draw=True):
        lmlist=[]
        if self.result.pose_landmarks:
            for id,lm in enumerate(self.result.pose_landmarks.landmark):
                h,w,c=img.shape
                cx=int(lm.x*w)
                cy=int(lm.y*h)
                lmlist.append(lm)
                cv2.putText(img,str(id),(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0))
        return img,lmlist
def main():
    ptime=0
    ctime=0
    vid=cv2.VideoCapture(0)
    detector=pose_detection()
    while True:
        check,img=vid.read()
        img=detector.Detect_Pose(img)
        img,lmlist=detector.find_pos(img)
        ctime=time.time()
        fps=1/(ctime-ptime)
        ptime=ctime
        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,0))
        cv2.imshow("image",img)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
if __name__=="__main__":
    main()
