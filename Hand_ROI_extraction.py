import cv2
import time

static_back = None

upper_left = (350, 50)
bottom_right = (600, 350)
    
video = cv2.VideoCapture(0) 

while True: 
    check, ROI = video.read() 
    ROI = cv2.flip(ROI,1)
    frame = ROI[50:350,350:600]
    
    r = cv2.rectangle(ROI, upper_left, bottom_right, (100, 50, 200), 5)
   
    # Converting color image to gray_scale image 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
  
    # Converting gray scale image to GaussianBlur 
    gray = cv2.GaussianBlur(gray, (21, 21), 0) 
  
    #static frame 
    if static_back is None: 
        static_back = gray 
        continue
  
    # Difference between static background and present frame
    diff_frame = cv2.absdiff(static_back, gray) 
  
    #threshold to see if theres a movement 
    thresh_frame = cv2.threshold(diff_frame, 15, 255, cv2.THRESH_BINARY)[1] 
    thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2) 
  
    # Find contour of moving object 
    cnts,_ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
  
    for contour in cnts: 
        if cv2.contourArea(contour) < 10000: 
            continue
  
        (x, y, w, h) = cv2.boundingRect(contour) 
        # make rectangle arround the moving object 
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        time.sleep(1)
        cv2.imwrite("1.png",frame)

    cv2.imshow("Threshold Frame", thresh_frame) 
    
    cv2.imshow("Color Frame", frame) 
    cv2.imshow("roi",ROI)
    
    key = cv2.waitKey(1) 
    
video.release() 
cv2.destroyAllWindows() 