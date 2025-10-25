import cv2 
import random
from ultralytics import yolo
from google.colab.patches import cv2_imshow


#colors for each class
yolo = YOLO("yolov8s.pt")
def getColors(cls_num)
    random seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range (3))

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

