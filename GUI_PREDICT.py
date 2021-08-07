import cv2
import numpy as np
from tkinter import *
from PIL import ImageGrab
from tensorflow.keras.models import load_model
model = load_model('model_hand.h5')

source = Tk() #Create the main window
source.resizable(0, 0)
source.title("FINAL YEAR PROJECT BLIND ASSISSTANT")
initx, inity = None, None
image_number = 0

def clear_source():
    global draw_area
    draw_area.delete("all")#Delete method for cleaning


def activate_event(event):
    global initx, inity
    draw_area.bind('<B1-Motion>', draw_lines)#Session has started and call draw_lines
    initx, inity = event.x, event.y


def draw_lines(event):
    global initx, inity
    x, y = event.x, event.y
    #Do the drawing
    draw_area.create_line((initx, inity, x, y), width=7, fill='black', capstyle=ROUND, smooth=True, splinesteps=12)
    initx, inity = x, y

def Recognize_Charac():
    global image_number
    filename = f'image_{image_number}.png'
    widget=draw_area
    #Get coordinates of canvas.
    x = source.winfo_rootx() + widget.winfo_x()
    y = source.winfo_rooty() + widget.winfo_y()
    x1 = x+widget.winfo_width()
    y1 = y+widget.winfo_height()
    #Get image by using grab() and crop it. Then save it.
    ImageGrab.grab().crop((x, y, x1, y1)).save(filename)

    charac = cv2.imread(filename, cv2.IMREAD_COLOR)#Read image
    make_gray=cv2.cvtColor(charac, cv2.COLOR_BGR2GRAY)#Convert in grayscale
    ret, th = cv2.threshold(make_gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)#Otsu threshold
    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]#extracting contours from image

    for cnt in contours:
        word_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
                     12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
                     23: 'X', 24: 'Y', 25: 'Z'}

        #Get bounding box
        x, y, w, h = cv2.boundingRect(cnt)
        #Create Rectangle
        cv2.rectangle(charac, (x, y), (x+w, y+h), (255, 0, 0), 1)
        top = int(0.05*th.shape[0])
        bottom = top
        left = int(0.05*th.shape[1])
        right = left
        #Extract the image ROI
        roi = th[y-top:y+h+bottom, x-left:x+w+right]
        #Resize ROI image

        img = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)



        #Reshape image to standart of our model
        img = img.reshape(1, 28, 28, 1)
        prediction = model.predict([img])[0]
        final = word_dict[np.argmax(prediction)]

        #Get the maximum values
        data = str(final)+'  '+str(int(max(prediction)*100))+'%'
        #Draw the screen
        cv2.putText(charac, data, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.imshow('Character', charac)#Show the result
    cv2.waitKey(0)

#Creating canvas
draw_area=Canvas(source, width=640, height=480, bg='white')
draw_area.grid(row=0, column=0, pady=2, sticky=W, columnspan=2)
#Mechanism to let you deal with event yourself
draw_area.bind('<Button-1>', activate_event)
#Add buttons and their functions
btn_save = Button(text="Recognize the Character", fg='black', command=Recognize_Charac)
btn_save.grid(row=2, column=0, pady=1, padx=1)
button_clear = Button(text="Clear Area", fg='black', command=clear_source)
button_clear.grid(row=2, column=1, pady=1, padx=1)

source.mainloop()# When our code is ready, this function will run.
