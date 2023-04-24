

""" 





videos = glob.glob('data/videos/*.*')
exit_flag =  False

yon = "horizontal"
roi =  0.5 
pedestrian_list = []
pedestrian_id = 1
count_up_to_down = 0
count_down_to_up = 0
count_left_to_right = 0
count_right_to_left = 0

for video in videos:
    if exit_flag: break
    video_name = video.split('/')[-1].split('.')[0]
    cam1 = cv2.VideoCapture(video)
    counter = 0
    while True:
        counter += 1
        ret, framex = cam1.read()
        if not ret: 
            print(f'Bitti {video}')
            break
        
        hh, ww, _ = framex.shape
        ratio = ww/hh
        width = 720
        frame = cv2.resize(framex, (width, int(width/ratio)))
        
        # Process the input image
        preprocessed_image, image_with_bbox, filtered_bboxes_obj = preprocess_image_and_detect_pedestrian(frame.copy())

        # Convert the preprocessed_image to a 3-channel image
        preprocessed_image_color = cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2BGR)
        

        try: fitered_bboxes = filtered_bboxes_obj['bbox'] 
        except: fitered_bboxes = []
        # Call the update_count function to update the counts
        count_up_to_down, count_down_to_up, count_left_to_right, count_right_to_left = counting_pedestrian(yon, roi, fitered_bboxes, frame)

        # Display the pedestrian count on the image
        count_text = f'Asagi: {count_up_to_down}, Yukari: {count_down_to_up}, Saga: {count_left_to_right}, Sola: {count_right_to_left}'
        cv2.putText(image_with_bbox, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Display the original, preprocessed, and images with bounding boxes side by side
        combined_image = cv2.hconcat([preprocessed_image_color, image_with_bbox])
        cv2.imshow('Comparison', combined_image)


        if counter % 5 == 0 and filtered_bboxes_obj:
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imname = f'data/bw/images/img{video_name}_{counter:03d}.jpg'
            # cv2.imwrite(imname, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])  # Set JPEG quality to 95
            print('Video name: ', video_name, 'Image Name : ', imname) 
            # Save the preprocessed_image_color to the 'images' folder
            preprocessed_image_color = cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2BGR)
            # cv2.imwrite(imname, preprocessed_image_color, [cv2.IMWRITE_JPEG_QUALITY, 95])  # Set JPEG quality to 95

            # Create and save Pascal VOC annotations for the current frame
            #create_pascal_voc_annotation(os.path.basename(imname), filtered_bboxes, frame.shape[1], frame.shape[0])
            
        key = cv2.waitKey(50) & 0xFF  # Changed from cv2.waitKey(0) to cv2.waitKey(1)
        if key == ord('q'):
            
            cap.release()
            cap = None
            cv2.destroyAllWindows()
            exit_flag = True
            break
         """
        
 
        
        
'''

def _preprocess_image(image, min_size=70, max_iou=0.7):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply threshold to highlight pedestrians
    _, thresh_image = cv2.threshold(blurred_image, 128, 255, cv2.THRESH_BINARY)

    # Apply morphological operations to refine the binary image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded_image = cv2.erode(thresh_image, kernel, iterations=1)
    dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)

    # Apply additional morphological operations to separate touching objects
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilated_image2 = cv2.dilate(dilated_image, kernel2, iterations=2)
    eroded_image2 = cv2.erode(dilated_image2, kernel2, iterations=2)

    # Detect contours in the binary image
    contours, _ = cv2.findContours(eroded_image2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



    # Draw bounding boxes around the detected contours and filter by size
    image_with_bbox = image.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= min_size and h >= min_size:  # Check if the bounding box meets the minimum size requirement
            current_bbox = (x, y, w, h)
            should_add = True

            if not is_valid_aspect_ratio(current_bbox):
                should_add = False

            for existing_bbox in filtered_bboxes:
                if bbox_iou(current_bbox, existing_bbox) > max_iou:
                    should_add = False
                    break

            if should_add:
                cv2.rectangle(image_with_bbox, (x, y), (x + w, y + h), (0, 255, 0), 2)
                filtered_bboxes.append(current_bbox)

    return dilated_image, image_with_bbox, filtered_bboxes

def detect_pedestrians(image, scale=1.1, win_stride=(2, 2), padding=(16, 16)):
    # Resize the image to make the detection process faster
    image_resized = imutils.resize(image, width=min(400, image.shape[1]))

    # Detect pedestrians in the image
    (rects, weights) = hog.detectMultiScale(image_resized, winStride=win_stride, padding=padding, scale=scale)

    # Convert the rectangles coordinates to the original image scale
    rects = [[x * (image.shape[1] / float(image_resized.shape[1])),
              y * (image.shape[0] / float(image_resized.shape[0])),
              w * (image.shape[1] / float(image_resized.shape[1])),
              h * (image.shape[0] / float(image_resized.shape[0]))] for (x, y, w, h) in rects]

    return rects
    '''
    
    
    
    
""" 
  for td in trdata:
            
            x_min, y_min, x_max, y_max, trackID, score, labelid = td[0].item(), td[1].item(), td[2].item(), td[3].item(), td[4].item(), td[5].item(), td[6].item()
            x_min, y_min, x_max, y_max = int(x_min * width), int(y_min * height), int(x_max * width), int(y_max * height)
            ww, hh = x_max - x_min, y_max - y_min
            
            # Convert coordinates from relative to absolute
            cw, ch = x_min + int(ww / 2), y_min + int(hh / 2)
            
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)
            cv2.circle(frame, (int(cw), int(ch)), 8, (0,255,0), 2)
            cv2.putText(frame, f'{int(trackID)}', (int(cw), int(ch)), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)
            # text 
            text = f"Label {LABELS[int(labelid)]}"
            text += f'\nScore: {score:.2f}'
            text += f'\nTrk delay: {time.monotonic() - time2:.2f}'
            time2=time.monotonic()
            alert =  True  # check_people_in_restricted_area(cw, ch, data, width, height)
            cv2.putText(frame, f'Det:{DETECTION_THRESHOLD:.2f} {len(trdata)} kisi, Blue:{BLUE_THRESHOLD:.2f}, Red:{RED_THRESHOLD:.2f} Tracker: {use_tracker} ', 
                            (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)
            # if (height * )
            for line in text.split('\n'):
                y_min += 30  
                cv2.putText(frame, line, (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1) # font, font_scale, color, thickness)
        frame_count += 1 
        # general info:   
        cv2.putText(frame, 'Model: ' + MODEL_PATH + ' Video:  ' + VIDEO_PATH, 
                    (5,25),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)
        
        # Distance info
        # cv2.putText(frame, distance_text, (width - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
             
             


###########################
# Normalize and batchify the image
def pre_process_image(image_np, type='normalized', input_size=(256,256)):
    if type=='normalized': 
        image = cv2.resize(image_np, input_size)
        image = np.expand_dims(image/255.0, axis=0).astype(np.float32)
    else:    
        image = np.expand_dims(image_np, axis=0).astype(np.uint8)
    return image

def create_feature_extractor(model_path="models/deep_sort/mars-small128.pb"):
    
    with tf.io.gfile.GFile(model_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

        sess = tf.compat.v1.Session()
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name="")

    return sess

def extract_features(sess, image, input_name="images:0", output_name="features:0"):
    features = sess.run(output_name, feed_dict={input_name: image[np.newaxis]})
    return features.flatten()
    
    
# Call the function to add new trackers
new_trackers, new_texts = add_new_trackers(trackers, detected_objects, iou_threshold=0.2, frame=frame)
trackers.extend(new_trackers)
textes.extend(new_texts)  




                det_line = f'{time.strftime("%d/%m/%Y %H:%M")} Alanda {len(trackers)} kişi tespit edildi.#'
                if write_control and det_line_pre != det_line:
                    cv2.imwrite(out_write_path, frame)
                    file = time.strftime("%d%m%Y")
                    with open(f'data/logs/log{file[:]}.csv', 'a') as ff:
                        wr = writer(ff)
                        wr.writerow([det_line+fileext])
                        det_lines.append(det_line+fileext)
                        det_line_pre = det_line
                        
                        
                        
                        

# Update the add_new_trackers() function
def add_new_trackers(existing_trackers, detected_objects, iou_threshold, frame):
    tt = time.monotonic()
    new_trackers = []
    new_texts = []
    height, width, _ = frame.shape
        
    for obj in detected_objects:
        # Convert coordinates from relative to absolute        
        for obj in detected_objects:
            recent_bbox = obj['bbox']
            add_new_tracker = True
            for tracker in existing_trackers:
                ok, tracker_bbox = tracker.update(frame)
                x, y, w, h = tracker_bbox
                tracker_bbox_converted = (y / height, x / width, (y + h) / height, (x + w) / width)
                iou = bbox_iou(recent_bbox, tracker_bbox_converted, width, height, bbox1_normalized=True, bbox2_normalized=True)
                if iou > iou_threshold:
                    add_new_tracker = False
                    break
                
        if add_new_tracker:
            y_min, x_min, y_max, x_max = recent_bbox
            x_min, y_min, x_max, y_max = int(x_min * width), int(y_min * height), int(x_max * width), int(y_max * height)
            ww, hh = x_max - x_min, y_max - y_min
            tracker = cv2.TrackerCSRT_create()
            ok = tracker.init(frame, (x_min, y_min, ww, hh))
            new_trackers.append(tracker)
            blue_percentage = calculate_blue_percentage(frame, (x_min, y_min, x_max, y_max))
            red_yellow_percentage = calculate_red_yellow_percentage(frame, (x_min, y_min, x_max, y_max))
            blue_red_yellow_per.append((blue_percentage, red_yellow_percentage))
            score = obj['score'] * 100
            text = f'{classes[0]} \n%{score:.0f} \nBlue:{blue_percentage:.2f}\nRed:{red_yellow_percentage:.2f}\nAddtime{time.monotonic()-tt:.2f}'
            
            textes.append(text)
                        
    return new_trackers, new_texts            
                             
 """
 
 
 
""" 
def preprocess_image_and_detect_pedestrian(image, blue_threshold, red_yellow__threshold, model, min_size=40, max_iou=0.5, maskSize=5, threshold_value=0.5, erode_iterations=1, dilate_iterations=1):
    blue_percentage = 0.0
    red_yellow_percentage = 1.0
    
    height, width, _ = image.shape
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    normalized_gray = cv2.normalize(blurred_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Apply threshold to highlight pedestrians
    threshold = int(threshold_value * 255)  # Convert threshold value to an integer between 0 and 255
    #_, thresh_image = cv2.threshold(blurred_image, threshold, 255, cv2.THRESH_BINARY)
    _, thresh_image = cv2.threshold(normalized_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    # Apply morphological operations to refine the binary image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded_image = cv2.erode(thresh_image, kernel, iterations=1)
    dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)
    
    # Apply additional morphological operations to separate touching objects
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilated_image2 = cv2.dilate(dilated_image, kernel2, iterations=2)
    eroded_image2 = cv2.erode(dilated_image2, kernel2, iterations=2)
    
    # Apply morphological operations to refine the binary image
    eroded_image2 = cv2.erode(eroded_image2, kernel, iterations=erode_iterations)
    dilated_image2 = cv2.dilate(eroded_image2, kernel, iterations=dilate_iterations)

    # Apply watershed algorithm to separate touching objects
    distance_transform = cv2.distanceTransform(eroded_image2, cv2.DIST_L2, maskSize)
    _, sure_fg = cv2.threshold(distance_transform, threshold_value * distance_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(dilated_image2, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)
            
    # Create a list to store the filtered bounding boxes
    filtered_bboxes = []
    filtered_bboxes_normalized = []
  
    # Draw bounding boxes around the detected contours and filter by size
    for i in range(2, np.max(markers) + 1):
        obj = {}
        points = np.where(markers == i)
        x, y, w, h = cv2.boundingRect(np.array([points[1], points[0]]).T)
        x2, y2 = x+w, y+h
        if model == "opencv":
            blue_percentage = calculate_blue_percentage(image, (x, y, x2, y2))
            red_yellow_percentage = calculate_red_yellow_percentage(image, (x, y, x2, y2))
            
        is_blue = blue_percentage < blue_threshold
        is_red_yellow = red_yellow_percentage > red_yellow__threshold
        print('Blue :', blue_percentage, is_blue , ' Red_Yel: ', red_yellow_percentage, is_red_yellow )
        
        
        if w >= min_size and h >= min_size and is_valid_aspect_ratio((x,y,w,h)) and is_blue and is_red_yellow:
            current_bbox = (x, y, w, h)
            obj['bbox'] = [ y/height, x/width, (y+h) / height, (x+w) / width ]
      
            should_add = True

            for existing_bbox in filtered_bboxes:
                iou = bbox_iou(current_bbox, existing_bbox, width, height)
                if iou > max_iou or is_bbox_inside(current_bbox, existing_bbox) or is_bbox_inside(existing_bbox, current_bbox):
                    should_add = False
                    break 
                

            if should_add:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(image, f'RT:{h/w:.2f}', (x,y+20), cv2.FONT_HERSHEY_COMPLEX, 0.8,(0,0,255), 1)
                cv2.putText(image, f'BL{blue_percentage:.2f}', (x,y+40), cv2.FONT_HERSHEY_COMPLEX, 0.8,(0,0,255), 1)
                cv2.putText(image, f'RE{red_yellow_percentage:.2f}', (x,y+60), cv2.FONT_HERSHEY_COMPLEX, 0.8,(0,0,255), 1)
                filtered_bboxes.append(current_bbox)
                obj['class'] = 0.0
                obj['score'] = 1.0
    
                filtered_bboxes_normalized.append(obj)

    return dilated_image, image, filtered_bboxes_normalized  #filtered_bboxes """