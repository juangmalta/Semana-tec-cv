#En este progama con una imagen nos genera dos filtros, uno en blanco y negro y otra estilo comic 
import cv2
import numpy as np

#Pasamos de parametro la imagen
img=cv2.imread("manuel.jpeg")

#Imagen blanco y negro
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray=cv2.medianBlur(gray,5)
edges=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,9)

#Imagen estilo comic
color=cv2.bilateralFilter(img,9,250,250)
cartoon=cv2.bitwise_and(color,color,mask=edges)

#Parte que muestra las imagenes
cv2.imshow("Image",img)
cv2.imshow("Edges",edges)
cv2.imshow("Cartoon", cartoon)
cv2.waitKey(0)
cv2.destroyAllWindows()
