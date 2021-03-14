import cv2
import time
from keras.models import model_from_json
import numpy as np

static_back = None

upper_left = (350, 50)
bottom_right = (600, 350)
    
video = cv2.VideoCapture(0) 

# load the model
json_file = open('model.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model_weights_1.h5")

pred_list = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']

while True: 
    check, frame = video.read() 
    frame = cv2.flip(frame,1)
    ROI = frame[50:350,350:600]
    
    r = cv2.rectangle(frame, upper_left, bottom_right, (100, 50, 200), 1)
   
    # Converting color image to gray_scale image 
    gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY) 
  
    # Converting gray scale image to GaussianBlur 
    gray = cv2.GaussianBlur(gray, (21, 21), 0) 
  
    #static frame 
    if static_back is None: 
        static_back = gray 
        continue
  
    # Difference between static background and present frame
    diff_ROI = cv2.absdiff(static_back, gray) 
  
    #threshold to see if theres a movement 
    thresh_ROI = cv2.threshold(diff_ROI, 15, 255, cv2.THRESH_BINARY)[1] 
    thresh_ROI = cv2.dilate(thresh_ROI, None, iterations = 2) 
  
    # Find contour of moving object 
    cnts,_ = cv2.findContours(thresh_ROI.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
  
    for contour in cnts: 
        if cv2.contourArea(contour) < 10000: 
            continue
        # make rectangle arround the moving object
        #(x, y, w, h) = cv2.boundingRect(contour)  
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        time.sleep(0.2)

        #save image
        gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (28,28), interpolation = cv2.INTER_CUBIC)
        cv2.imwrite("Output.png",resized)

        # predict the model
        keras_model= model.predict(resized[np.newaxis, :, :, np.newaxis])
        pred = np.argmax(keras_model)
        #print(pred)
        print(pred_list[pred+1])

    cv2.imshow("Test", frame) 
    cv2.imshow("Thresh", thresh_ROI)
    
    key = cv2.waitKey(1) 
    
video.release() 
cv2.destroyAllWindows() 