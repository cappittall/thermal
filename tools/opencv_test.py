import cv2
import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'tools/'))
                
from tools.pedesterian_detector import CV2PedestrianDetector
from tools.cv_detection import bind_opencv_result_on_frame 
detector = CV2PedestrianDetector()
# Open the video file
video = cv2.VideoCapture('data/videos/1.mp4')
BLUE_THRESHOLD = 1.0 ## maximum mavi oranı 
RED_THRESHOLD = 0.0 ## Minimum kırmızı oranı 
MODEL = "opencv"
# Loop through the frames
width = 960

while True:
    # Read a frame
    ret, frame_ = video.read()

    # Break the loop if we reach the end of the video
    if not ret:
        break
    ratio = frame_.shape[0]/frame_.shape[1]
    
    frame = cv2.resize(frame_, (width, int(width * ratio) ))
    
    preprocessed_image, image_with_bbox, detected_objects = \
                detector.preprocess_image_and_detect_pedestrian(frame.copy(), BLUE_THRESHOLD, RED_THRESHOLD, MODEL)
    frame = bind_opencv_result_on_frame(preprocessed_image, image_with_bbox, frame)
    # Display the frame
    cv2.imshow('Opencv detection', frame)

    # Check if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video and close all windows
video.release()
cv2.destroyAllWindows()
