import cv2
import ultralytics
import cvzone
import mediapipe as mp
from time import time

#####################################
offsetpercentagew = 10
offsetpercentageh = 20
confidence =0.8
camwidth,camheight  = 640,480
floatingpoint = 6
save=True
BlurThreshold = 35 #larger the value more the focus
outputFolderPath = "D:/data_collection"
debug =False
classID = 1 #0 IS FAKE and 1 Is real, basically we are labelling the data through this parameter

#####################################

from cvzone.FaceDetectionModule import FaceDetector

cap = cv2.VideoCapture(2)
cap.set(3,camwidth)
cap.set(4,camheight)
detector = FaceDetector()
while True:
    success, img = cap.read()
    img, bboxs = detector.findFaces(img,draw = False)
    imgout = img.copy()
    listBlur = [] #this will indicate whether the faces are blur or not
    listInfo = [] #this will have normalised values and the class name for the label text file
    if bboxs:
        for bbox in bboxs:
            x,y,w,h = bbox["bbox"]
            score  = float(bbox["score"][0])

            if score>confidence:
                offsetw = (offsetpercentagew / 100) * w
                offseth = (offsetpercentageh / 100) * h
                x = x - offsetw
                w = w + offsetw * 2
                y = y - offseth * 3
                h = h + offseth * 3

                if x < 0: x = 0
                if y < 0: y = 0
                if w < 0: w = 0
                if h < 0: h = 0

                # find blurriness
                imgface = img[y:y + h, x:x + w]
                cv2.imshow("face", imgface)
                blurvalue = int(cv2.Laplacian(imgface, cv2.CV_64F).var())
                if blurvalue > BlurThreshold:
                    listBlur.append(True)
                else:
                    listBlur.append(False)

                # normalise values
                ih, iw, _ = img.shape
                xc, yc = x + w / 2, y + h / 2
                xcn, ycn = round((xc / iw), floatingpoint), round((yc / ih), floatingpoint)
                wcn, hcn = round((w / iw), floatingpoint), round((h / ih), floatingpoint)

                if xcn > 1: xcn = 1
                if ycn > 1: ycn = 1
                if wcn > 1: wcn = 1
                if hcn > 1: hcn = 1
                listInfo.append(f"{classID} {xcn} {ycn} {wcn} {hcn}\n")

                cv2.rectangle(imgout, (int(x), int(y), int(w), int(h)), (255, 0, 0), 3)
                cvzone.putTextRect(imgout, f'score:{int(score * 100)}% Blue:{blurvalue}', (x, y - 20), scale=2,
                                   thickness=3)

                if debug:
                    cv2.rectangle(img, (int(x), int(y), int(w), int(h)), (255, 0, 0), 3)
                    cvzone.putTextRect(imgout, f'score:{int(score * 100)}% Blue:{blurvalue}', (x, y - 20), scale=2,
                                       thickness=3)
            if save:
                if all(listBlur) and listBlur != []:
                    timeNow = time()
                    timeNow = str(timeNow).split('.')
                    timeNow = timeNow[0] + timeNow[1]
                    cv2.imwrite(f"{outputFolderPath}/{timeNow}.jpg", img)
                    for info in listInfo:
                        f = open(f"{outputFolderPath}/{timeNow}.txt", 'a')
                        f.write(info)
                        f.close()





    cv2.imshow("Image", imgout)
    cv2.waitKey(1)
