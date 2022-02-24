
import numpy as np
import cv2
from imutils.object_detection import non_max_suppression
import face_recognition

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


# open webcam video stream
cap = cv2.VideoCapture(0)

# the output will be written to output.avi
out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640,480))

#file location 
daniel_image = face_recognition.load_image_file("D:/Course Material Sem VIII/Final Project II/Implementation Code/SVM-based-multi-view-face-recognition-using-HOG-Histogram-of-Oriented-Gradients-technique-master/daniel.jpeg")
daniel_face_encoding = face_recognition.face_encodings(daniel_image)[0]

known_faces = [daniel_face_encoding]

#  Initialize some variables
face_locations = []
face_encodings = []
face_names = []

badan = True 



while(True): 
    # Capture frame-by-frame
    ret, frame = cap.read()        
    # resizing 
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    if badan == True:
        boxes, weights = hog.detectMultiScale(frame, winStride=(16,16),padding=(32,32), scale=1.05, finalThreshold=2)  

        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        pick = non_max_suppression(boxes, probs=None,overlapThresh=0.65)
        for (xA, yA, xB, yB) in pick:
            # display the detected boxes in the colour picture
            cv2.rectangle(frame, (xA, yA), (xB, yB),(0, 0, 255), 4)

            #calculate distance between object and camera
            k = 2*((xB-xA) + (yB-yA))

            if (k in range (1130,1264)):
                print("Jarak 3 meter")
            elif (k in range (826,918)):
                print("Jarak 4 meter")
            elif (k in range (594,660)):
                print("Jarak 5 meter")
            elif (k in range (486,546)):
                print("Jarak 6 meter")
            elif (k in range (452,510)):
                print("Jarak 7 meter")
            elif (k in range (432,450)):
                print("Jarak 8 meter")
            else :
                print("Jarak lebih dari 8 meter")

            



    # detec human face after system succesfull to detect human body
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match 
        match = face_recognition.compare_faces(
            known_faces, face_encoding, tolerance= 0.6)

        # set false for badan when the system find a face
        name = None
        if match[0]:
            badan = False
            name = "Daniel"
        
        
        face_names.append(name)

    # give a Label to the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if not name:
            continue

        # make a rectangle
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # make label 
        cv2.rectangle(frame, (left, bottom - 35),(right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 2)


    
    # Write the output video 
    out.write(frame.astype('uint8'))
    cv2.imshow('frame',frame) 
    #spesifikasi kamera = 60fps
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

out.release()

cv2.destroyAllWindows()
cv2.waitKey(1)  