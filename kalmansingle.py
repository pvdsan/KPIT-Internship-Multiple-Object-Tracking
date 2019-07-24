import cv2

import numpy as np

 

 

 

 

class KalmanFilter:

 

    kalman = cv2.KalmanFilter(8,4)

    kalman.measurementMatrix = np.eye(4,8)

    kalman.transitionMatrix = np.array([[1,0,0,0,1/30,0,0,0],[0,1,0,0,0,1/30,0,0],[0,0,1,0,0,0,1/30,0],[0,0,0,1,0,0,0,1/30],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]],np.float32)

    kalman.processNoiseCov =np.array([[1,0,0,0,1/30,0,0,0],[0,1,0,0,0,1/30,0,0],[0,0,1,0,0,0,1/30,0],[0,0,0,1,0,0,0,1/30],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]],np.float32)

    mp = np.zeros((4,1), np.float32) # tracked / prediction

    tp = np.zeros((4,1), np.float32)

    meas=[]

    pred=[]

 

 

 

cap=cv2.VideoCapture("video2.mp4")

ok,frame=cap.read()

 

NumOfTrackers=2

 

BoundaryList=[]

TrackerArray=[]

KalmanList=[]

   

for i in range(0,NumOfTrackers):

    bbox = cv2.selectROI("ROI Selector", frame, False)   

    BoundaryList.append(bbox)

    p1 = (int(BoundaryList[i][0]), int(BoundaryList[i][1]))

    p2 = (int(BoundaryList[i][0] + BoundaryList[i][2]), int(BoundaryList[i][1] + BoundaryList[i][3]))

    h = int(BoundaryList[i][2])

    w= int(BoundaryList[i][3]) 

    kalmanOBJ=KalmanFilter()

   

    mp=[p1[0],p1[1],h,w]

       

    kalmanOBJ.meas.append(tuple(mp))

    kalmanOBJ.pred.append(tuple(mp))

    KalmanList.append(kalmanOBJ)

   

        

    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

 

 

 

for k in BoundaryList:

    tracker=cv2.TrackerKCF_create()

    ok=tracker.init(frame,k)

    TrackerArray.append(tracker)

 

    

while True:

    ok,img=cap.read()

       

    i=0

    

    for TrackIter in TrackerArray:

       

        ok,BoundaryList[i]=TrackIter.update(img)

        p1 = (int(BoundaryList[i][0]), int(BoundaryList[i][1]))

        p2 = (int(BoundaryList[i][0] + BoundaryList[i][2]), int(BoundaryList[i][1] + BoundaryList[i][3]))

        h = int(BoundaryList[i][2])

        w= int(BoundaryList[i][3])

        p = np.array([np.float32(p1[0]),np.float32(p1[1]),np.float32(h),np.float32(w)])

        KalmanList[i].mp=np.array(p.reshape(4,1))

        print(KalmanList[i].mp)

        pp = np.array(KalmanList[i].mp)

        KalmanList[i].kalman.correct(pp)

        tp=KalmanList[i].kalman.predict()

        print(tp)

             

        """predcoord=(int(tp[0]),int(tp[1]),int(tp[2]),int(tp[3]))

       

        KalmanList[i].pred.append(predcoord)

       

        cv2.rectangle(img,KalmanList[i].pred[i],p2, (255, 0, 0), 2, 1)"""

     

        i=i+1

       

        

      

        

    cv2.namedWindow("Tracking")

    cv2.moveWindow("Tracking", 20, 20)

    cv2.imshow("Tracking", img)

        # Exit if ESC pressed

    k = cv2.waitKey(30) & 0xff

    if k == 27:

        break
