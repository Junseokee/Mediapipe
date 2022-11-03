import cv2
import os

vidcap=cv2.VideoCapture('C:/Users/MIS1/Desktop/mediapipe/squat3.avi')

cnt=0
os.mkdir('C:/Users/MIS1/Desktop/mediapipe/squat4/')

while(vidcap.isOpened()):
  ret, image = vidcap.read()
  if ret == False:
    break

  if(int(vidcap.get(1)) % 20 == 0): # 25프레임당 1장씩 저장 == 1초
    if cnt % 10 ==0:
      print('Save img number:' + str(int(cnt)))
    cv2.imwrite('C:/Users/MIS1/Desktop/mediapipe/squat4/squat_%d.jpg'%cnt, image)
    cnt+=1

vidcap.release()
#cv2.destroyAllWindows()