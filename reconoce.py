import cv2
import os
import numpy as np
import csv

dataPath = '/home/abi/git_workspace/Semana-tec-cv/Data' #Cambia a la ruta donde hayas almacenado Data
imagePaths = os.listdir(dataPath)
attendance = [False] * len(imagePaths)
print('imagePaths=',imagePaths)


face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Leyendo el modelo
face_recognizer.read('modeloLBPHFace.xml')
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('/home/abi/git_workspace/Semana-tec-cv/Videos/juanZ.mp4') #Video try

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

while True:
	ret,frame = cap.read()
	if ret == False: break
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.medianBlur(gray,25)
	edges=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,9)
	auxFrame = gray.copy()
	#color=cv2.bilateralFilter(imagePaths,9,250,250)
	cartoon=cv2.bitwise_and(gray,gray,mask=edges)

	faces = faceClassif.detectMultiScale(gray,1.3,5)

	for (x,y,w,h) in faces:
		rostro = auxFrame[y:y+h,x:x+w]
		rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
		result = face_recognizer.predict(rostro)

		cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
		# LBPHFace
		if result[1] < 70:
			cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
			cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
			attendance[result[0]]=True

		else:
			cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
			cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)

	cv2.imshow('frame',frame)
	cv2.imshow('cartoon',cartoon)
	#cv2.imshow("Cartoon", cartoon)
	k = cv2.waitKey(1)
	if k == 27:
		break

#Creararchivo attendnce
with open('attendance.csv', 'w') as file:
	for index, alumno in enumerate(imagePaths):
		file.write("{}, {}".format(alumno, "Present" if attendance[index] else "Absent"))
		file.write('\n')

file.close()
cap.release()
cv2.destroyAllWindows()
