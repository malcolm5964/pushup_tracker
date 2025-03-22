import cv2
import numpy as np
import mediapipe as mp
import pose_estimation as pm
import time
import paho.mqtt.client as mqtt
from firestore_manager import FirestoreManager
import os
import json

# Global variables
count = 0                  # Number of successful pushups
direction = 0              # 0: going down, 1: going up
attempt_count = 1          # Number of total attempts
bad_form_images = []       # List to store paths of saved bad form images
current_attempt_saved = False  # Track if we've saved a bad form image for the current attempt
counting = False           # Whether we're in counting mode
ready_hold_time = None     # Time when ready position was first detected
last_status = None         # Last user status
timer_started = False      # Whether the session timer has started
timer_start_time = None    # When the timer started
mqtt_client = None         # MQTT client
firestore_mgr = None       # Firestore manager

def setup_mqtt():
    """Initializes and returns an MQTT client."""
    global mqtt_client
    broker_address = "172.20.10.4"
    client = mqtt.Client("PushupCounter") # Create client with ID "PushupCounter"
    
    try:
        client.connect(broker_address, 1883) # Connect to broker on standard port
        client.loop_start() # Start background thread to handle network traffic
        mqtt_client = client
        return True
    except Exception as e:
        print(f"MQTT connection error: {e}")
        return False

def setup_camera():
    """Initializes the video capture object."""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)  # Keep resolution balanced for speed
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)  # Request 30 FPS (Mediapipe will process as fast as possible)
    return cap 
    
def detect_form_issues(elbow, shoulder, hip):
    """Detect specific form issues and returns the issue description if found."""
    if elbow > 90 and elbow < 160:
        return "Elbow angle incorrect"
    elif hip < 160:
        return "Hip angle incorrect - back not straight"

def save_bad_form(img, issue_type, attempt_num):
    """Save images of bad form for later review."""
    # Create directory if it doesn't exist
    os.makedirs("bad_form", exist_ok=True)
    
    # Generate unique filename with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"bad_form/attempt_{attempt_num}_{issue_type}_{timestamp}.jpg"
    
    # Save the image
    cv2.imwrite(filename, img)
    print(f"Saved bad form image: {filename}")
    
    return filename

def send_to_firebase():
    #code to send saved bad form images to firebase
    return
    
def update_count(elbow, shoulder, hip, img):
    """Determines the feedback message and updates the count based on the angles.
    Also detects form issues and uploads to Firestore if necessary."""
    global count, direction, attempt_count, timer_started
    global timer_start_time, bad_form_images, current_attempt_saved, mqtt_client

    current_time = time.time()

    if hip < 160 and current_attempt_saved == False:
        form_issue = detect_form_issues(elbow, shoulder, hip) 
        current_attempt_saved = True
        save_bad_form(img, form_issue, attempt_count)

    #Check going DOWN
    if direction == 0:
        #down_abnormal =  mqtt_client.subscribe("pushup/status")
        down_abnormal = False
        if elbow <= 90 and not down_abnormal:
            direction = 1

            # Start the 60-second timer on first down movement if not already started
            if not timer_started and mqtt_client:
                timer_started = True
                timer_start_time = current_time
                mqtt_client.publish("pushup/status", "Start", qos=2)
                print("Timer started! 60 seconds countdown begins.")
        
        elif down_abnormal and not current_attempt_saved:
            form_issue = detect_form_issues(elbow, shoulder, hip)
            current_attempt_saved = True
            save_bad_form(img, form_issue, attempt_count)

        return

    #Check going UP
    if direction == 1:
        #up_abnormal = mqtt_client.subscribe("pushup/status")
        up_abnormal = False
        if elbow > 160 and not up_abnormal:
            count += 1
            attempt_count += 1
            direction = 0
            mqtt_client.publish("pushup/status", "Push up counted", qos=2)
            print(f"Count: {count}")

        elif up_abnormal and not current_attempt_saved:
            current_attempt_saved = True
            form_issue = detect_form_issues(elbow, shoulder, hip)
            save_bad_form(img, form_issue, attempt_count)
            direction = 0
            current_attempt_saved = False

        return
        
        
def check_valid_pose(lmList):
    """Checks if a valid person pose is detected."""
    valid_pose = False
    if len(lmList) >= 25:  # Make sure we have enough landmarks
        key_points = [11, 13, 15, 23, 25]  # Shoulder, elbow, wrist, hip, knee
        if all(point < len(lmList) for point in key_points):
            valid_pose = True
    return valid_pose


def check_ready_position(elbow, shoulder, hip):
    """Checks if person is in the UP position and ready to start counting."""
    global ready_hold_time, counting, mqtt_client

    in_up_position = (elbow > 155 and shoulder > 40 and hip > 155)
        
    if in_up_position:
        # Start or continue the ready timer
        if ready_hold_time is None:
            ready_hold_time = time.time()

        # If held for 1.5 seconds, start counting
        if time.time() - ready_hold_time > 1.5:
            print("You may start!")
            # Publish MQTT message when user is in position
            if mqtt_client:
                mqtt_client.publish("pushup/status", "User in position", qos=2)
                print("Published: User in position")

            counting = True
            return True  # Return ready_hold_time, counting=True
    else:
        ready_hold_time = None
        print("Get in UP Position")
    
    return False
    
#Check if user exist
def check_user(lmList):
    """Checks if a valid person pose is detected."""
    global last_status, mqtt_client, timer_started, timer_start_time, counting, count, direction

    valid_user = False
    if len(lmList) >= 25:  # Make sure we have enough landmarks
        key_points = [11, 13, 15, 23, 25]  # Shoulder, elbow, wrist, hip, knee
        if all(point < len(lmList) for point in key_points):
            valid_user = True   

    # Track status change
    current_status = "User detected" if valid_user else "No user detected"
    status_changed = (last_status is None or current_status != last_status)

    # Only publish when status actually changes
    if mqtt_client and status_changed:
        # Clear any queued messages by sending with higher QoS
        mqtt_client.publish("pushup/status", current_status, qos=2)
        print(f"Status CHANGED to: {current_status}")

    if not valid_user:
        timer_started = False
        timer_start_time = None
        counting = False
        count = 0
        direction = 0
        mqtt_client.publish("pushup/status", "End", qos=2)

    last_status = current_status
    return valid_user
    
def check_60s(current_time):
    """Check if 60 seconds have elapsed and handle session end if needed."""
    global timer_started, timer_start_time, bad_form_images, count, attempt_count
    global ready_hold_time, current_attempt_saved, mqtt_client, counting, direction

    if timer_started:
        elapsed_time = current_time - timer_start_time
        remaining_time = max(0, 60 - elapsed_time)
        #print(f"Remaining time: {remaining_time}")

        if remaining_time <= 0:
            mqtt_client.publish("pushup/status", "End", qos=2)
            
            # Prepare session stats
            session_stats = {
                "timestamp": time.time(),
                "total_pushups": count,
                "total_attempts": attempt_count,
                "success_rate": (count / max(1, attempt_count)) * 100,
                "session_duration": 60  # seconds
            }    

            # Send bad form images to Firebase
            send_to_firebase(session_stats)

            # Reset timer and session variables
            timer_started = False
            timer_start_time = None
            counting = False
            count = 0
            attempt_count = 0
            direction = 0
            bad_form_images = []
            ready_hold_time = None
            current_attempt_saved = False

        return

def calculate_fps(prev_time):
    """Calculate frames per second."""
    cur_time = time.time()
    fps = 1 / (cur_time - prev_time)
    print(f"FPS: {fps:.2f}")
    return cur_time, fps
    

def main():
    global count, direction, attempt_count, bad_form_images, current_attempt_saved
    global counting, ready_hold_time, last_upload_time, last_status
    global timer_started, timer_start_time, mqtt_client, firestore_mgr

    # Initialize variables
    count = 0
    direction = 0
    attempt_count = 0
    bad_form_images = []
    current_attempt_saved = False
    counting = False
    ready_hold_time = None
    last_upload_time = None
    last_status = None
    timer_started = False
    timer_start_time = None

    #Setup
    cap = setup_camera()
    detector = pm.poseDetector()
    setup_mqtt() #Set up mqtt
    firestore_mgr = FirestoreManager(credential_path="firebase-credentials.json") # Initialize Firestore manager

    # Ensure bad_form directory exists
    os.makedirs("bad_form", exist_ok=True)

    prev_time = time.time()  # Initialize FPS timer
    
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        current_time = time.time()
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)

        valid_user = check_user(lmList) #Check if user exist

        if not valid_user:
            continue

        # Get key angles
        elbow = detector.findAngle(img, 11, 13, 15)
        shoulder = detector.findAngle(img, 13, 11, 23)
        hip = detector.findAngle(img, 11, 23, 25)

        # Check if person is in UP position (starting position)
        if not counting:
            position_ready = check_ready_position(elbow, shoulder, hip)

        # If counting is active, perform pushup detection
        if counting:
            update_count(elbow, shoulder, hip, img.copy())
            check_60s(current_time)
            
        # FPS Calculation
        #prev_time, fps = calculate_fps(prev_time)
        
        cv2.imshow('Pushup Counter', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
