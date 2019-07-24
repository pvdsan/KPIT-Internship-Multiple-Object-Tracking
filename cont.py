

# Required libraries

import cv2

import numpy as np

import math

import sys

 

 
 

# Global Variables

 

 

dt=1/20                    # Time for the next frame(Optional)

 

FrameCount=0               # Frame Number to stop at designated frame to annotate

 

TrackerArray=[]            # Global List of all trackers

 

NumOfTrackers=2            # Number of trackers to be initialised

  

NetObjectNumber=0          # Net number of objects after merging and splitting

 

Thresh=60                  # Threshold pixel distance for splitting and merging

 

BboxList=[]                # Global list of annotated boundaries

 

FrameRef=0                 # Frame Number used for saving frames in folder

 

  

# Opening Cap wrt to video

cap=cv2.VideoCapture("video2.mp4")

"""

Kalman Class, used to asociate the kalman model and the KCF tracker as a single object

 

"""

# Scope of class started

#---------------------------------------------------------------------------------------

 

 

class KalmanFilter:

 

# Class Constructor to create seperate instance of atrributes in every case   

    def __init__(self):

   

    

        # State Transition Matrix

        self.A=np.array([[1,0,0,0,1,0,0,0],[0,1,0,0,0,1,0,0],[0,0,1,0,0,0,1,0],[0,0,0,1,0,0,0,1],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]],np.float32)

        # Observation Matrix

        self.H=np.eye(4,8)

 

        # Initial Covariance matrix

        self.P=np.eye(8,8)*10000

     

        # Estimated measurement error covariance matrix

        self.R=np.eye(4,4)*12000

         

        # State vector with x,y,w,h and their derivatives 

        self.CurrentState=np.zeros((8,1),np.float32)

 

        # Process Noise Conversion Matrix

        self.PNC=np.ones((8,1),np.float32)*0.0003

   

        # Measurement Noise conversion matrix

        #( Since measurement is absolute and error free, we do not use this here)

        self.MNC=np.ones((4,1),np.float32)*0.0003

   

        # Vector of rectangular coordinates of bounding box,(lower right and upper left point)

        self.RectCoord=np.zeros((4,1),np.float32)

     

        # Tracker object initialised (Inbuilt OpenCv)

        self.tracker=cv2.TrackerKCF_create()

   

        #Measurement matrix having current measurement of the bounding box, x,y,w,h

        self.CurrentMeasurement=np.zeros((4,1),np.float32)

       

        # TrackId of the Object(Used for merging and splitting)

        self.TrackId=-1

   

   

# Class function to store annotated boundary boxes in list

    def GetBoundaryBox(self,frame):

    

        global BboxList

    
        # Inbuilt OpenCv function to annotate an ROI(object of interest)

 

        bbox1 = cv2.selectROI("Select Objects to track", frame, False)

        # Initialising Measurement values of the object in the CurrentMeasuremnt Matrix

        self.CurrentMeasurement[0]=float(bbox1[0])

       

        self.CurrentMeasurement[1]=float(bbox1[1])

               

        self.CurrentMeasurement[2]=float(bbox1[2])

       

        self.CurrentMeasurement[3]=float(bbox1[3])

    
        # Initialising Rectangular Corrdinates Matrix

        self.RectCoord[0]=float(bbox1[0])

       

        self.RectCoord[1]=float(bbox1[1])

       

        self.RectCoord[2]=float((bbox1[0]+bbox1[2]))

 

        self.RectCoord[3]=float((bbox1[1]+bbox1[3]))

    
        #Add bounding box coordinates to the global list of boundaries

        BboxList.append(bbox1)

       
        # Draw bounding box for refernece and visualization

        p1 = (int(bbox1[0]), int(bbox1[1]))

       

        p2 = (int(bbox1[0] + bbox1[2]), int(bbox1[1] + bbox1[3]))

       

        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

       

        

# Class function to Update the Process noise conversion matrix based on velocity       

    def UpdatePNC(self):

       

        

        # Bottom right quandrant of OpenCV

        if(self.CurrentState[4]>0 and self.CurrentState[5]>0):

            self.PNC[0]=abs(self.CurrentState[4]/10)

            self.PNC[1]=abs(self.CurrentState[5]/10)

       

        

        # Top left quadrant of OpenCv

        elif(self.CurrentState[4]<0 and self.CurrentState[5]<0):

       

            self.PNC[0]=(-1*abs(self.CurrentState[4]/10))

            self.PNC[1]=(-1*abs(self.CurrentState[5]/10))

           

        # Top right quadrant of OpenCv

        elif(self.CurrentState[4]>0 and self.CurrentState[5]<0):

            self.PNC[0]=abs(self.CurrentState[4]/10)

            self.PNC[1]=(-1*abs(self.CurrentState[5]/10))

           

        else:

           

            self.PNC[0]=(-1*abs(self.CurrentState[4]/10))

            self.PNC[1]=abs(self.CurrentState[5]/10)

 

        

        

# Class function to update all measurement and state values based on latest bounding box given by the tracker   

    def GetStateVector(self,bbox):

       

        # Get latest bounding box measurements

        self.CurrentState[0]=float(bbox[0]) #x

       

        self.CurrentState[1]=float(bbox[1]) #y

 

        self.CurrentState[2]=float(bbox[2]) #w

 

        self.CurrentState[3]=float(bbox[3]) #h

       

        # Find derivatives by subracting the previous values stored in the CurrentMeasurement matrix

        self.CurrentState[4]=(float(bbox[0])-self.CurrentMeasurement[0])

 

        self.CurrentState[5]=(float(bbox[1])-self.CurrentMeasurement[1])

       

        self.CurrentState[6]=(float(bbox[2])-self.CurrentMeasurement[2])

       

        self.CurrentState[7]=(float(bbox[3])-self.CurrentMeasurement[3])

 

        # update CurrentMeasurement with latest values

        self.CurrentMeasurement[0]=float(bbox[0])

       

        self.CurrentMeasurement[1]=float(bbox[1])

       

        self.CurrentMeasurement[2]=float(bbox[2])

       

        self.CurrentMeasurement[3]=float(bbox[3])

       

 

# Class function to perform time update and state prediction steps of the Kalman filter

    def ProcessEqu(self):

       

        

        # State prediction step

        xp= np.add(self.PNC,np.dot(self.A,self.CurrentState))

                

        # Innovation step       

        z1=np.dot(self.H,xp)

        zk=np.subtract(self.CurrentMeasurement,z1)

               
        # Covariance prediction step

        Pp= np.dot(np.dot(self.A,self.P),np.transpose(self.A))

       
        # Innovation covariance step

        S= np.add(np.dot(np.dot(self.H,Pp),np.transpose(self.H)),self.R)

       
        # Kalman Gain Calculation matrix

        K=np.dot(np.dot(Pp,np.transpose(self.H)),np.linalg.inv(S))

       
        #Covariance Update

        self.P=np.dot(np.subtract(np.eye(8,8),np.dot(K,self.H)),Pp)

       

        #Update Current State based on the Kalman gain

        self.CurrentState=np.add(xp,np.dot(K,zk))

    
    
        # Update the rectangle coordinates of the tracker      

        self.RectCoord[0]=self.CurrentState[0]

        self.RectCoord[1]=self.CurrentState[1]

        self.RectCoord[2]=self.CurrentState[0]+self.CurrentState[2]

        self.RectCoord[3]=self.CurrentState[1]+self.CurrentState[3]

 

""" Class scope is completed"""


#----------------------------------------------------------------------------------------------------------------------------------------------------
  

# Global function to initialise the KalmanObjects and store in Tracker Array

      

def BoundTrackers():

   

    global FrameCount   

    global NumOfTrackers 

    global TrackerArray

   
# Initial video commencement to stop and annotate whenever required   

    while(True):

        ok,frame1=cap.read()

        FrameCount=FrameCount+1

        cv2.imshow("Select Frame to Annotate",frame1)

        # When frame of interest found

        if(cv2.waitKey(0) & 0xFF==ord('s')):

            break

   

    for i in range(0,NumOfTrackers):

       

        kalman=KalmanFilter()                     #Create kalman object

        kalman.tracker=cv2.TrackerKCF_create()    #Initialise tracker attribute to KCF

        kalman.GetBoundaryBox(frame1)             #Associate object with the initial annotated measurements

        kalman.tracker.init(frame1,BboxList[i])   # Initialise object with the respective annotated boundary

        TrackerArray.append(kalman)               # Add object to global TrackerArray

   

 

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#Global function to compute distance two points        

def ComputeDistance(x1,y1,x2,y2):

    return math.sqrt(pow((x2-x1),2)+pow((y2-y1),2))

 
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

 
# Global function to find the first point of merged object(upper left point)

def FindPoint1(List):

       

    minx=List[0].RectCoord[0]

    for i in range(0,len(List)):

        if(List[i].RectCoord[0]<minx):

            minx=List[i].RectCoord[0]

   

    miny=List[0].RectCoord[1]

    for i in range(0,2):

        if(List[i].RectCoord[1]<miny):

            miny=List[i].RectCoord[1]

 

    point=(minx,miny)

   

    return(point)

 
# Global function to find the second point of merged object(lower right point)

   

def FindPoint2(List):


    maxx=List[0].RectCoord[2]

    for i in range(0,len(List)):

        if(List[i].RectCoord[2]>maxx):

            maxx=List[i].RectCoord[2]

   

    maxy=List[0].RectCoord[3]

    for i in range(0,2):

        if(List[i].RectCoord[3]>maxy):

            maxy=List[i].RectCoord[3]

 

    point=(maxx,maxy)

   

    return(point)   

    

#--------------------------------------------------------------------------------------------------------------------------------------------------

# Grouping tracked objects based o track ID

"""

If any two or more tracked object and closer to each other than the threshold

specified globally, they are considered candidates to be merged and asigned

the same TrackId

"""

 

#Graph Coloring concept is used

 

def AssignTrackId():

   

    global Thresh

    global NetObjectNumber

   

    IdNumber=1

    for i in range(0,NumOfTrackers):

        if(TrackerArray[i].TrackId==-1):

            TrackerArray[i].TrackId=IdNumber

            IdNumber=IdNumber+1           

        for j in range(i+1,NumOfTrackers):


            x1=(TrackerArray[i].RectCoord[0]+TrackerArray[i].RectCoord[2])/2

            y1=(TrackerArray[i].RectCoord[1]+TrackerArray[i].RectCoord[3])/2           

            x2=(TrackerArray[j].RectCoord[0]+TrackerArray[j].RectCoord[2])/2

            y2=(TrackerArray[j].RectCoord[1]+TrackerArray[j].RectCoord[3])/2

           
            d=ComputeDistance(x1,y1,x2,y2)

            if(d<Thresh):        

                TrackerArray[j].TrackId=TrackerArray[i].TrackId

 

    NetObjectNumber=IdNumber-1


#------------------------------------------------------------------------------------------------------------------------------------------------- 


# Function to implement merging on the objects falling under same TrackID

""" Earlier we have defined TrackiDs based on the proximity of two objects,

Based on that we merge the two bounding boxes of the objects to show as a single entity

 

"""

 

def MergeId(image):

   

    global NetObjectNumber


    for i in range(1,NetObjectNumber+1):

    
        List=[]

        for trk in TrackerArray:

           if(trk.TrackId==i):

               List.append(trk)

          
        if(len(List)>1):

            k=FindPoint1(List)

            l=FindPoint2(List)

            cv2.rectangle(image,k,l,(255,0,0),2)

        else:

            cv2.rectangle(image,(List[0].RectCoord[0],List[0].RectCoord[1]),(List[0].RectCoord[2],List[0].RectCoord[3]),(255,0,0),2)


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 

# Gloabal Function associated with resetting the TrackerId so that we use

 

def RestartId():

    for i in range(NumOfTrackers):

        TrackerArray[i].TrackId=-1
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#Main Function scope

 
 

BoundTrackers()   #Initialise trackers and their boundaries


# Tracking video    

while(cap.isOpened()):

      

    ok,image=cap.read()

    FrameRef=FrameRef+1

    for i in range(0,NumOfTrackers):

        ok,bbox=TrackerArray[i].tracker.update(image) #Tracker updates new position of the object

        if(ok==True):  # If successfully tracked,

            #Kalman Calculations

             TrackerArray[i].GetStateVector(bbox) 

             TrackerArray[i].ProcessEqu()

             TrackerArray[i].UpdatePNC()

        else:

            # if tracker fails

            #extrapolate coordinates based on their previous derivatives

        
            TrackerArray[i].CurrentState[0]=TrackerArray[i].CurrentState[0]+TrackerArray[i].CurrentState[4]

            TrackerArray[i].CurrentState[1]=TrackerArray[i].CurrentState[1]+TrackerArray[i].CurrentState[5]

            TrackerArray[i].CurrentState[2]=TrackerArray[i].CurrentState[2]+TrackerArray[i].CurrentState[6]

            TrackerArray[i].CurrentState[3]=TrackerArray[i].CurrentState[3]+TrackerArray[i].CurrentState[7]

           

            TrackerArray[i].RectCoord[0]=TrackerArray[i].CurrentState[0]

            TrackerArray[i].RectCoord[1]=TrackerArray[i].CurrentState[1]

            TrackerArray[i].RectCoord[2]=TrackerArray[i].CurrentState[0]+TrackerArray[i].CurrentState[2]

            TrackerArray[i].RectCoord[3]=TrackerArray[i].CurrentState[1]+TrackerArray[i].CurrentState[3]

                   
    AssignTrackId()

    MergeId(image)

    cv2.imshow("Object Tracking",image)

    name="frame%d.jpg"%FrameRef

    cv2.imwrite(name,image)

    RestartId()

 
    if cv2.waitKey(0) & 0xFF==27:

        break

  

cap.release()

cv2.destroyAllWindows()   