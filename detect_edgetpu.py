import base64
from datetime import datetime

from typing import List
import warnings
import os
import time
from csv import writer
import json

from multiprocessing import Queue
from fastapi import FastAPI, File, Request, Response

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import FileResponse
import sys

from tools.cv_detection import bind_opencv_result_on_frame, clip_bbox

sys.path.append(os.path.join(os.getcwd(), "tools/"))

from pydantic import BaseModel
import numpy as np

import cv2
import time

# Edge tpu libraries
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

from sort.sort import Sort
from tools.pedesterian_detector import CV2PedestrianDetector
from tools.pedestrian_counter import LineCounter, LineCounterAnnotator
from dataclasses import dataclass
import supervision
print("supervision.__version__:", supervision.__version__)
from supervision.tools.detections import Detections, BoxAnnotator
# from supervision.tools.line_counter import LineCounter, LineCounterAnnotator
from supervision.geometry.dataclasses import Point
from supervision.draw.color import ColorPalette

BASE_PATH  = os.getcwd()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

warnings.filterwarnings('ignore')

### Previous constants  ###
SERVER="localhost:8500"

# Initialize the process and queues
detect_process = None
result_queue = Queue()

VIDEO_PATH=""
processes = []
REDETECT=True
DETECTION_THRESHOLD = 0.4
distance = None
classes = [""]
scores=[]
frame_count = 0
alert = False
cap=trdata=interpreter=tracker=None
line_counter=line_annotator = box_annotator= None

blue_red_yellow_per=[]
tracker_histrory = None
#Sort 

det_line_pre = ""
time2=0
try: 
    det_lines = open(f'data/logs/log{time.strftime("%d%m%Y")}.csv', 'r' ).read().splitlines()
except:
    det_lines = []

if not os.path.exists('data/logimg'): os.mkdir('data/logimg')
if not os.path.exists('data/logs'): os.mkdir('data/logs')

class LineValuesAndCheckboxes(BaseModel):
    lineValues: List[int]
    radioValues: List[bool]
    selectedVideo: str
    selectedModel: str
    start: bool
    use_tracker: bool
           
@app.get("/")
def read_root(request:Request=None):
    try: 
        cap.release()
        cap = None
    except: pass

    videos={ i:i for i in os.listdir('data/videos/')}
    models={ i:i for i in os.listdir('models/edges/') if i.endswith('.tflite')}
    if os.path.exists('data/initial.txt'): 
        initial_pos = [int(value) for value in open('data/initial.txt', 'r').read().splitlines()]
    else:initial_pos=[50,50,40] # 0: dikey 1: yatay 2,3: boş 4: DETECTION_THRESHOLD
    # Invoke the WSGI application function and return the response
    return templates.TemplateResponse("index.html", {"request": request, "videos": videos, "models": models, 'pos': initial_pos })

@app.post('/pedestal-tflite-model5x')
async def pedestal_tflite_model_test5x(data:LineValuesAndCheckboxes, request:Request=None):
    #print('Data..', data)

    DETECTION_THRESHOLD = data.lineValues[2] / 100
    BLUE_THRESHOLD = data.lineValues[3] / 100
    RED_THRESHOLD = data.lineValues[4] / 100
    MIN_BOX_SIZE = data.lineValues[6] 
    yon = "horizontal" if data.radioValues[1] else "vertical"
    roi = data.lineValues[data.radioValues.index(True)] / 100 
    # screen image size 
    width = 960
    height = 720
    desired_frame_rate = 24  # Set your desired frame rate

    # detected bbox =  x: cx - offset , y: cy - offset, x2: cx + offset, y2: cy + offset 
    offset = 10
    
    # use tracker or use detection ever single frame.
    use_tracker = data.use_tracker
    
    # global variables
    global cap, interpreter, VIDEO_PATH, MODEL_PATH, frame_count, \
            tracker, tracker_histrory, alert, det_line_pre, trdata, time2, \
            line_counter, line_annotator, box_annotator, opencv_detector, \
            skip_frames, frame_number, inference_size
    
    # stoping loop setups
    if not data.start:
        cap.release()
        cap = None
        trdata = None
        frame_count = 0
        time2=0
        tracker = None
        tracker_histrory = None
        return None
    
    if cap is None:
        # Initial setup & read initial setups in order to set initials
        with open('data/initial.txt', 'w') as f: f.write('\n'.join(map(str, data.lineValues)))
        VIDEO_PATH = "data/videos/" + data.selectedVideo
        MODEL_PATH = "models/edges/" + data.selectedModel
        cap = cv2.VideoCapture(VIDEO_PATH)
        original_frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        skip_frames = int(original_frame_rate / desired_frame_rate)
        frame_number = 0
        print('skip_frames', skip_frames,' Frame nr: ', frame_number )
        
        #
        if not data.selectedModel == "opencv": 
            interpreter = make_interpreter(MODEL_PATH)
            interpreter.allocate_tensors()
            inference_size = input_size(interpreter)
            
        # get tracker hystory. Inıt tracker
        tracker_histrory = data.lineValues[5] 
        tracker = Sort(max_age=tracker_histrory , min_hits=0, iou_threshold=.30)
        
        # counter 
        LINE_START = Point(0, int(height * roi)  ) if yon=="horizontal" else Point(int(width * roi), 0 )
        LINE_END = Point(width, int(height * roi)  ) if yon=="horizontal" else Point(int(width * roi), height )
        
        # create line and instance of BoxAnnotator
        line_counter = LineCounter(start=LINE_START, end=LINE_END )
        line_annotator = LineCounterAnnotator(thickness=1, text_thickness=1, text_scale=1)
        box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1, text_thickness=1, text_scale=1)
        opencv_detector = CV2PedestrianDetector()
        
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, im = cap.read()
    frame_number += skip_frames
    
    if success:
        video_ratio = im.shape[1]/im.shape[0]
        frame = cv2.resize(im, (width, int(width/video_ratio) ))
        height, width, _ = frame.shape
                
        # Loop in here every time if use _tracker is False. 
        if not use_tracker or frame_count % tracker_histrory == 0 :

            # trackers.clear()                              
            if data.selectedModel == "opencv":
                preprocessed_image, image_with_bbox, detected_objects = \
                    opencv_detector.preprocess_image_and_detect_pedestrian(im.copy(), BLUE_THRESHOLD, RED_THRESHOLD, data.selectedModel, min_size= MIN_BOX_SIZE)
                                   
                detections = []
                for obj in detected_objects:
                    y_min, x_min, y_max, x_max = obj['bbox']
                    x_min, y_min, x_max, y_max = clip_bbox(frame.shape, (x_min, y_min, x_max, y_max))
                    detections.append(np.array([x_min, y_min, x_max, y_max, obj['score'], obj['class'] ]))
            
            else:
                # prepare image before model input   
                if "thermal3" in data.selectedModel:
                    # convert image to black & white 
                    preprocessed_image = opencv_detector.preprocess_image_and_detect_pedestrian(im.copy(), BLUE_THRESHOLD, RED_THRESHOLD, data.selectedModel, min_size= MIN_BOX_SIZE)
                    preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2BGR)
                else: 
                    preprocessed_image = im.copy() 
                                   
                input_data = cv2.resize(preprocessed_image, inference_size)
                # prediction
                run_inference(interpreter, input_data.tobytes())
                objs = get_objects(interpreter, DETECTION_THRESHOLD )
                ## 
                image_with_bbox = input_data.copy()
                print("Objs : ", objs)
                detections = []
                for obj in objs:
                    x1,y1,x2,y2 = obj.bbox
                    sqrm2 = (x2-x1) * (y2 - y1) / 1000
                    if sqrm2 > 110 : continue
                    cv2.rectangle(image_with_bbox, (x1, y1), (x2, y2),(0,0,255),2)
                    cv2.putText(image_with_bbox, f'{obj.score * 100:.0f}%', (obj.bbox.xmin, obj.bbox.ymin + 20), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                    cv2.putText(image_with_bbox, f'{sqrm2:.0f} m2', (obj.bbox.xmin, obj.bbox.ymin + 60), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                    detections.append([obj.bbox.xmin / inference_size[0] , obj.bbox.ymin / inference_size[1], 
                                        obj.bbox.xmax / inference_size[0],  obj.bbox.ymax / inference_size[1], 
                                        obj.score, obj.id])
                
                
                
        # bing input image and detection image on right bottom                
        frame = bind_opencv_result_on_frame(preprocessed_image, image_with_bbox, frame)
        # prepare detections if not any, assign empty numpy array.
        if not detections: detections = np.empty((0,7))
        # update tracker.
        print("Detections: ", detections)
        trdata = tracker.update(np.array(detections))
        
        if trdata.any():
            xyxy = []
            confidence = []
            class_id = []
            tracker_id = []
            labels = []
            for x_min, y_min, x_max, y_max, trackID, score, labelid in trdata:
                # Calculate the center point (cx, cy)
                x_min, y_min, x_max, y_max = x_min * width,  y_min * height, x_max * width, y_max * height
                sqrm2 = int(((x_max-x_min) * (y_max-y_min)) / 1000)
                cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2           
                xyxy.append([cx - offset, cy - offset, cx + offset, cy + offset])  
                confidence.append(score)
                class_id.append(int(labelid))
                tracker_id.append(trackID)
                labels.append(f"#{int(trackID)},%{score * 100:.0f}^{sqrm2:.0f}")
            if len(xyxy) > 0 :
                detections = Detections(
                            xyxy=np.array(xyxy).astype(np.float32),
                            confidence=np.array(confidence).astype(np.float32),
                            class_id=np.array(class_id).astype(int),
                            tracker_id=np.array(tracker_id).astype(int))
                
                # labels=[f"#{int(trackID)},{score} Insan"  for  x_min, y_min, x_max, y_max, trackID, score, labelid in trdata]
                # updating line counter
                line_counter.update(detections=detections)
                # annotate and display frame
                frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
                line_annotator.annotate(frame=frame, line_counter=line_counter, write_in_out=False)  
    
        text = f'Yukari: {line_counter.in_count} Asagi: {line_counter.out_count}' 
        text2 = f'video: {data.selectedVideo} Model: {data.selectedModel}'
        cv2.putText(frame, text, (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.putText(frame, text2, (30, height -  20), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        is_success, img_buffer = cv2.imencode(".png", frame)
        if is_success:
            img_base64 = base64.b64encode(img_buffer).decode("utf-8")
            response_data = {
                "image": img_base64,
                "det_lines": det_lines,
                "base_path": BASE_PATH
            }

            response_json = json.dumps(response_data)
            return Response(response_json, media_type="application/json")
        
        
    else:
        cap.release()
        cap = None
        return None

@app.get("/image/{image_name}")
async def serve_image(image_name: str):
    image_path = f"{BASE_PATH}/data/logimg/{image_name}.jpg"
    return FileResponse(image_path, media_type="image/jpeg")

