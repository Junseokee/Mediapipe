import calculate_angle
import cv2
import mediapipe as mp
import numpy as np

# biceps curl

max = int(input('갯수를 입력하세요 : '))

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
# Curl conuter variables 54분
counter = 0
rcount = 0
stage = None


with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        
        # Detect stuff and render
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make detection
        results = pose.process(image)
        
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract Landmarks 43분부터
        try:
            landmarks = results.pose_landmarks.landmark
            if counter < max:
                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Calculate angle
                angle = calculate_angle(shoulder,elbow,wrist)
                # Visualize angle
                cv2.putText(image, str(angle),
                            tuple(np.multiply(elbow, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA
                                )
                # Curl counter Logic -> angle이 160도 이상이면 이완(down) -> 다운상태일때 30도 미만이면 up
                if angle > 160:
                    stage = "down"
                if angle < 30 and stage == 'down':
                    stage = 'up'
                    counter +=1
                    print(counter)
            elif counter == max:
                
                if rcount != counter:
                    # Get coordinates
                    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                    # Calculate angle
                    angle = calculate_angle(shoulder,elbow,wrist)
                    # Visualize angle
                    cv2.putText(image, str(angle),
                                tuple(np.multiply(elbow, [640, 480]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA
                                    )
                    # Curl counter Logic -> angle이 160도 이상이면 이완(down) -> 다운상태일때 30도 미만이면 up
                    if angle > 160:
                        stage = "down"
                    if angle < 30 and stage == 'down':
                        stage = 'up'
                        rcount +=1
                        print(rcount)


        except:
            pass
        
        # Render curl counter
        # setup status box
        cv2.rectangle(image, (0,0), (225,73), (245, 117,16),-1)
        
        # Rep Data
        cv2.putText(image, 'REPS', (15,12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        if counter == max:
            cv2.putText(image, str(rcount),
                (10,60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2,  cv2.LINE_AA)
        else:
            cv2.putText(image, str(counter),
                    (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2,  cv2.LINE_AA)
        # Rep Data 1시간
        cv2.putText(image, 'STAGE', (65,12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stage,
                (60,60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2,  cv2.LINE_AA)
        
        # Render detections 
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),#joint color
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)# bone color
                                )


        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        if counter & rcount == max:
            break
    cap.release()
    cv2.destroyAllWindows()