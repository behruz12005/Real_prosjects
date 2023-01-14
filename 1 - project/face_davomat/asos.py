#pylint:disable=no-member

import numpy as np
import cv2 as cv
import pandas as pd
import time

from datetime import date

# today() to get current date
todays_date = date.today()

df = pd.read_csv('Davomad.csv')
name = list(df['ism'].values)

haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = ['image','behruz']
features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

# img = cv.imread(r"C:\Users\behru\OneDrive\Desktop\kaskad_detection\data\images1\17_1050_1673603433.046439.png")

cap = cv.VideoCapture(0)

while True:
    _,img = cap.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Detect the face in the image
    faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)
    cv.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (255,0,0), thickness=80)
    cv.putText(img, 'Kameriga qarang', (85, 30), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 0), thickness=5)
    for (x,y,w,h) in faces_rect:
        faces_roi = gray[y:y+h,x:x+w]

        label, confidence = face_recognizer.predict(faces_roi)


        print(f'Label = {people[label]} with a confidence of {confidence}')
        cv.rectangle(img, (x, y), (x+w, y+h), (0,255,0), thickness=2)
        cv.putText(img, str(people[label]), (x, y - 5), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)

        data = list(df[df['ism'] == people[label]]['voqt'])
        print(data)
        try:
            if confidence <= 100  and data[0] == 'None':
                df.loc[df['ism'] == people[label], 'voqt'] = todays_date
            elif confidence > 100 and data[0] != 'None' and people[label] in name:
                cv.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (50, 50, 255), thickness=80)
                cv.putText(img, 'Sen ruyxatga olingansan', (85, 30), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 0),thickness=5)
                print(img.shape)
        except:
            continue




    cv.imshow('Detected Face', img)

    if cv.waitKey(1) == ord('q'):
        break
df.to_csv('Davomad.csv',index = False)
print(df)