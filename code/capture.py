# biceps curl
import cv2
import mediapipe as mp
import numpy as np

dir_path = "C:/Users/MIS1/Desktop/squat/"
video = ("C:/Users/MIS1/Desktop/yangssame.mp4")

video_split = video.split('/')[-1:]
name = video_split[0].split('.')[0]
cap = cv2.VideoCapture(video)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
fps = cap.get(cv2.CAP_PROP_FPS)
cnt = 0
counter = 0
stage = None


def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 


with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),#joint color
                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)# bone color
                            )

        
        try:
            landmarks = results.pose_landmarks.landmark
            k = cv2.waitKey(1)
            if k == 117:
                img_captured = cv2.imwrite(dir_path+'up/'+name+'_%d.jpg'%cnt, image)
                cnt+=1
            if k == 100:
                img_captured = cv2.imwrite(dir_path+'down/'+name+'_%d.jpg'%cnt, image)
                cnt+=1

        except Exception as e:
            pass
        
    
        cv2.imshow('Capture image', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()