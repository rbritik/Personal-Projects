import cv2 as cv
import numpy as np
video = cv.VideoCapture(0)
face_cascade = cv.CascadeClassifier(r'C:\Users\Acer\Downloads\data_science\Computer Vision\Opencv\haar-cascade-files-master\haarcascade_frontalface_alt.xml')
smile_cascade = cv.CascadeClassifier(r'C:\Users\Acer\Downloads\data_science\Computer Vision\Opencv\haar-cascade-files-master\haarcascade_smile.xml')
while True:
    ok, frame = video.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face_rect = face_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 4)
    cnt = 1
    keyPressed = cv.waitKey(1)
    for x,y,w,h in face_rect:
        frame = cv.rectangle(frame, (x,y), (x+w, y+h), (0,0,0),2)
        smile_rect = smile_cascade.detectMultiScale(gray, scaleFactor = 1.8, minNeighbors=18)
        for x,y,w,h in smile_rect:
            frame = cv.rectangle(frame, (x,y), (x+w, y+h),(0,255,0),2)
            print("Image",str(cnt)," saved")
            img_path = r'C:\Users\Acer\Pictures\Saved Pictures\img'+str(cnt)+'.jpg'
            cv.imwrite(img_path,frame)
            cnt+= 1
            if cnt>=2:
                break
    cv.imshow("live",frame)
    if (keyPressed & 0xFF == ord('q')):
        break
video.release()
cv.destroyAllWindows()
