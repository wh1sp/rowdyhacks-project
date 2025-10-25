import cv2 
import random
from ultralytics import YOLO

#colors for each class
yolo = YOLO("yolov8x.pt")
def getColors(cls_num):
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range (3))


#dont really know what face_cascade is for but it seems important
#face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml')


video_cap = cv2.VideoCapture(0)

frame_count = 0 

#actual processing
while True:
    ret, frame = video_cap.read()
    if not ret:
        print("cant read the frame")
        break
    result = yolo.track(frame, stream=True)


    for result in result: 
        class_names = result.names
        for box in result.boxes:
            if box.conf[0] > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cls = int(box.cls[0])
                class_name = class_names[cls]

                conf = float(box.conf[0])
                color = getColors(cls)
                 
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{class_name} {conf:.2f}",
                    (x1, max(y1 - 10,20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


#timeout break
    if frame_count < 200:
        cv2.imshow('frame 1', frame)
    else:
        break
   # cv2.imshow('frame 1', frame)
    frame_count += 1
#exit conditions
'''
    if cv2.waitKey(1) == ord('q'):
        break
'''
#increment the framecount for keeping the video updated
#    frame_count += 1


video_cap.release()
