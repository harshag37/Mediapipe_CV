from typing import Mapping
import cv2
import mediapipe as mp
import time
class HandTracking():
    def __init__(self,mode=False,maxhands=1,detectioncon=0.5,trackcon=0.5):
        self.mode=mode
        self.maxhands=maxhands
        self.detectioncon=detectioncon
        self.trackcon=trackcon
        self.mphands=mp.solutions.hands
        self.Hands=self.mphands.Hands(self.mode,self.maxhands,self.detectioncon,self.trackcon)
        self.mpdraw=mp.solutions.drawing_utils
    def detect_hands(self,img,draw=True):
        imgrgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.result=self.Hands.process(imgrgb)
        if draw:
            if self.result.multi_hand_landmarks:
                for handlms in self.result.multi_hand_landmarks:
                    self.mpdraw.draw_landmarks(img,handlms,self.mphands.HAND_CONNECTIONS)
        return img
    def find_pos(self,img,handno=0,draw=True):
        lmlist=[]
        if self.result.multi_hand_landmarks:
            for handlms in self.result.multi_hand_landmarks:
                for id,ln in enumerate(handlms.landmark):
                    lmlist.append(ln)
                    h,w,c=img.shape
                    cx=int(ln.x*w)
                    cy=int(ln.y*h)
                    cv2.putText(img,str(id),(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0))
        return img,lmlist

def main():
    ptime=0
    ctime=0
    vid=cv2.VideoCapture(0)
    detector=HandTracking()
    while True:
        check,img=vid.read()
        img=detector.detect_hands(img)
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
