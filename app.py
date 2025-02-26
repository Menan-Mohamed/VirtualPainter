import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
hand_draw = mp.solutions.drawing_utils

imgCanvas = np.zeros((720,1288,3),np.uint8)

xp,yp = None , None
draw = 0
shape = [1,0,0]
color = (0,0,250)
thickness = 10

while True:

    st, img = cap.read()
    frame_height, frame_width, _ = img.shape
    img = cv2.flip(img, 1)
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    coordlist = []
 
    
    if result.multi_hand_landmarks :
            hand = result.multi_hand_landmarks[0]
            hand_draw.draw_landmarks(img,hand,mp_hands.HAND_CONNECTIONS)

            for id, lm in enumerate(hand.landmark):

                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                coordlist.append([id, cx, cy])

            x1,y1 = coordlist[8][1:]

            thumb = coordlist[4][1] < coordlist[2][1]
            index = coordlist[8][2] < coordlist[6][2]
            middle = coordlist[12][2] < coordlist[10][2]
            ring = coordlist[16][2] < coordlist[14][2]
            pinky = coordlist[20][2] < coordlist[18][2]

            if not thumb and index and not middle and not ring and not pinky :
                draw = 1
            else :
                draw = 0
                if 0<x1<80 and 0<y1<80:
                    thickness = 10
                    color = (0,0,250)
                    shape = [1,0,0]
                elif 0<x1<80 and 80<y1<160:
                    thickness = 20
                    color = (0,0,0)
                    shape = [1,0,0]
                elif 0<x1<80 and 161<y1<242:
                    thickness = 10
                    color = (0,0,250)
                    shape = [0,1,0]
                elif 0<x1<80 and 243<y1<323:
                    thickness = 10
                    color = (0,0,250)
                    shape = [0,0,1]
                xp,yp = None , None

            if draw == 1:

                if xp is None and yp is None :
                    xp, yp = x1, y1  
                if shape[0] == 1:
                    cv2.line(imgCanvas,(xp,yp),(x1,y1),color,thickness)
                elif shape[1] == 1:
                    cv2.circle(imgCanvas,(x1,y1),100,color)
                elif shape[2] == 1:
                    cv2.rectangle(imgCanvas, (x1+100, y1+50), (x1, y1), color)
                xp,yp = x1,y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)

    imgCanvas = cv2.resize(imgCanvas, (frame_width, frame_height))
    imgInv = cv2.resize(imgInv, (frame_width, frame_height))

    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    em = cv2.imread('shapes/brush.png')  
    em = cv2.resize(em, (80,80))
    img[0:80, 0:80] = em

    em2 = cv2.imread('shapes/eraser.png')  
    em2 = cv2.resize(em2, (80,80))
    img[81:161, 0:80] = em2

    em3 = cv2.imread('shapes/circle.png')  
    em3 = cv2.resize(em3, (80,80))
    img[162:242, 0:80] = em3

    em4 = cv2.imread('shapes/rectangle.png')  
    em4 = cv2.resize(em4, (80,80))
    img[243:323, 0:80] = em4

    cv2.imshow('virtual painter',img)
    

    if cv2.waitKey(30) & 0xff == ord('x'):
        break
    
cap.release()
cv2.destroyAllWindows()
