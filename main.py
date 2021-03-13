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
static_back = None
upper_left = (350, 50)
bottom_right = (600, 350)

def show_frame():
    
    _, frame = video.read()
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame) 
    """
    #new code
    check, frame = video.read() 
    frame = cv2.flip(frame,1)
    ROI = frame[50:350,350:600]
    r = cv2.rectangle(frame, upper_left, bottom_right, (100, 50, 200), 1)
    gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY) 

    img = Image.fromarray(gray)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)
    """

show_frame() 

tk.Label(window,text="Display Results",font=(None, 40)).grid(row=2,column=0)
tk.Label(window,text="Place Holder for results",font=(None, 40)).grid(row=2,column=0)
#canvas.pack()

window.mainloop()  #Starts GUI