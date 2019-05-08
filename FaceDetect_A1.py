import numpy as np
import cv2
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (224, 224)

face_cascade = cv2.CascadeClassifier('modelo/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture('arquivos/video2.mp4')

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MPEG')
out = cv2.VideoWriter('output_face.avi',fourcc, 25.0, (1920,1080))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
       # frame = cv2.blur(frame, ( 100, 100))#cv2.flip(frame,0)
        img = frame
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # write the flipped frame
        
        #here we go
        faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        )
        for (x,y,w,h) in faces:
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
        #here we stop
        
        out.write(img)

        cv2.imshow('frame',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()