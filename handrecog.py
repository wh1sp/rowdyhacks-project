import cv2 
import random
from ultralytics import YOLO
import time
import pyautogui
#import keyboard

left_pressed = False
right_pressed = False



#colors for each class
yolo = YOLO("yolov8x.pt")
def getColors(cls_num):
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range (3))

#video capture
video_cap = cv2.VideoCapture(0)


#dont really know what face_cascade is for but it seems important
#face_cascade = cv2.CascadeClassifier(cv2.datsya.haarcascades + "haarcascade_frontalface_default.xml')
#load the hand cascade
hand_cascade = cv2.CascadeClassifier('haarcascade_hand.xml')



frame_count = 0
#actual processing
while True:
    ret, frame = video_cap.read()
    if not ret:
        print("cant read the frame")
        break


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hands = hand_cascade.detectMultiScale(gray, 1.1,4)
    #leave this for now
    hand_positions = []
    height, width, _ = frame.shape
    mid_point = width // 2
    #frame = cv2.flip(frame, 1)

#exit conditions

    if cv2.waitKey(1) == ord('q'):
        break

    #hand positional stuff
    for (x1,y1,x2,y2) in hands:

        center_x = x1 + x2 // 2
        center_y = y1 + y1 // 2
        hand_positions.append(center_x)
        cv2.rectangle(frame, (x1,y1), (x1+x2, y1+y2), (255,0,0),2)
        cv2.circle(frame, (center_x, center_y),5,(0,255,0),-1)

    move_left = move_right = False

#debug stuff

    #overlay = frame.copy()
    #cv2.rectangle(overlay, (0, 0), (mid_point, height), (255, 0, 0), -1)
    #cv2.rectangle(overlay, (mid_point, 0), (width, height), (0, 0, 255), -1)
    #frame = cv2.addWeighted(overlay, 0.1, frame, 0.9, 0)



    if len(hand_positions) >= 1:
        avg_x = sum(hand_positions) / len(hand_positions)
        if avg_x < mid_point:
            move_left = True
        else:
            move_right = True

    if move_left:
        if not left_pressed:
            pyautogui.keyDown('a')
            left_pressed = True
        if right_pressed:
            pyautogui.keyUp('d')
            right_pressed = False
            cv2.putText(frame, "LEFT", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if move_right:
        if not right_pressed:
            pyautogui.keyDown('d')
            right_pressed = True
        if left_pressed:
            pyautogui.keyUp('a')
            left_pressed = False
            cv2.putText(frame, "RIGHT", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    else:
        # middle
        if left_pressed:
            pyautogui.keyUp('a')
            left_pressed = False
        if right_pressed:
            pyautogui.keyUp('d')
            right_pressed = False
            cv2.putText(frame, "CENTER / NONE", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        '''
        keyboard.press("a")
    else:
        keyboard.release("a")
'''


    cv2.line(frame, (mid_point, 0), (mid_point, height), (0, 255, 0), 2)
    cv2.imshow('frame1', frame)


    #full tracking
    '''
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
'''

#timeout break


    #cv2.imshow('frame 1', frame)
    #frame_count += 1



#increment the framecount for keeping the video updated
    frame_count += 1
    time.sleep(0.01)

pyautogui.keyUp('a')
pyautogui.keyUp('d')
video_cap.release()
cv2.destroyAllWindows()

