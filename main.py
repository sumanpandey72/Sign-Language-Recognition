import numpy as np
import cv2
import tkinter as tk
#from tkinter import *
from PIL import Image, ImageTk
import time

#Set up GUI
window = tk.Tk()  #Makes main window
#window = Tk()
#window.title("Reading Sign Language")
window.wm_title("Reading Hand Signals")
window.config(background="#FFFFFF")
#canvas = Canvas(height=1000, width=1000, bg='#B1DDC6', highlightbackground="#B1DDC6")

#Title in window


#Graphics window
#imageFrame = canvas.create_image(100,100)
imageFrame = tk.Frame(window, width=600, height=500)
imageFrame.grid(row=1, column=0, padx=10, pady=2)


#Capture video frames
lmain = tk.Label(imageFrame)
lmain.grid(row=1, column=0)

#start video feed code
video = cv2.VideoCapture(0)

upper_left = (350, 50)
bottom_right = (600, 350)

def gray_live_video():
    static_back = None
    while True:
        _, frame = video.read()
        frame = cv2.flip(frame, 1)
        ROI = frame[50:350,350:600]
        r = cv2.rectangle(frame, upper_left, bottom_right, (100, 50, 200), 1)
        gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if static_back is None: 
            static_back = gray 
            continue

        diff_ROI = cv2.absdiff(static_back, gray) 

        thresh_ROI = cv2.threshold(diff_ROI, 15, 255, cv2.THRESH_BINARY)[1] 
        thresh_ROI = cv2.dilate(thresh_ROI, None, iterations = 2) 

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

        return gray

def show_frame():
    img = Image.fromarray(gray_live_video())
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame) 

show_frame() 

tk.Label(window,text="Display Results",font=(None, 40)).grid(row=2,column=0)
tk.Label(window,text="Place Holder for results",font=(None, 40)).grid(row=3,column=0)
#canvas.pack()

window.mainloop()  #Starts GUI