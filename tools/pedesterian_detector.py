import cv2
import numpy as np

class CV2PedestrianDetector:
    def __init__(self):
        pass

    def bbox_iou(self, bbox1, bbox2, width, height, bbox1_normalized=True, bbox2_normalized=True):
    # Modify the bbox_iou function to accept both normalized and non-normalized coordinates
        y1_min, x1_min, y1_max, x1_max = bbox1
        y2_min, x2_min, y2_max, x2_max = bbox2

        if bbox1_normalized:
            x1_min, y1_min, x1_max, y1_max = int(x1_min * width), int(y1_min * height), int(x1_max * width), int(y1_max * height)
        if bbox2_normalized:
            x2_min, y2_min, x2_max, y2_max = int(x2_min * width), int(y2_min * height), int(x2_max * width), int(y2_max * height)

        x_intersection = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_intersection = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        intersection_area = x_intersection * y_intersection

        area_bbox1 = (x1_max - x1_min) * (y1_max - y1_min)
        area_bbox2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area_bbox1 + area_bbox2 - intersection_area
        

        try:
            iou = intersection_area / union_area
        except: iou = 0

        return iou


    def clip_bbox(self, image_shape, bbox):
        height, width, _ = image_shape
        x_min, y_min, x_max, y_max = bbox
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(width, x_max)
        y_max = min(height, y_max)
        return x_min, y_min, x_max, y_max

    def calculate_red_yellow_percentage(self, image, bbox):
        x_min, y_min, x_max, y_max = bbox
        cropped = image[y_min:y_max, x_min:x_max]
        if cropped.size == 0:
            return -.99
        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

        # Red color range
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        # Yellow color range
        lower_yellow = np.array([20, 50, 50])
        upper_yellow = np.array([40, 255, 255])

        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        mask_red_yellow = cv2.bitwise_or(mask_red1, mask_red2)
        mask_red_yellow = cv2.bitwise_or(mask_red_yellow, mask_yellow)

        red_yellow_pixels = cv2.countNonZero(mask_red_yellow)
        total_pixels = cropped.size // 3

        return red_yellow_pixels / total_pixels

    def calculate_blue_percentage(self, image, bbox):
        x_min, y_min, x_max, y_max = bbox
        cropped = image[y_min:y_max, x_min:x_max]
        if cropped.size == 0:
            return -.99
        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])

        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_pixels = cv2.countNonZero(mask)
        total_pixels = cropped.size // 3

        return blue_pixels / total_pixels        

    def is_bbox_inside(self, bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        return x1 >= x2 and y1 >= y2 and (x1 + w1) <= (x2 + w2) and (y1 + h1) <= (y2 + h2)

    def is_valid_aspect_ratio(self, bbox, min_ratio=0.4, max_ratio=1.8):
        x, y, w, h = bbox
        aspect_ratio = h / w

        return min_ratio <= aspect_ratio <= max_ratio

    def convert_to_grayscale_and_blur(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        return blurred_image

    def threshold_image(self, blurred_image, threshold_value):
        normalized_gray = cv2.normalize(blurred_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        _, thresh_image = cv2.threshold(normalized_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh_image

    def apply_morphological_operations(self, thresh_image, erode_iterations=1, dilate_iterations=1):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        eroded_image = cv2.erode(thresh_image, kernel, iterations=1)
        dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)
        
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        dilated_image2 = cv2.dilate(dilated_image, kernel2, iterations=2)
        eroded_image2 = cv2.erode(dilated_image2, kernel2, iterations=2)
        
        eroded_image2 = cv2.erode(eroded_image2, kernel, iterations=erode_iterations)
        dilated_image2 = cv2.dilate(eroded_image2, kernel, iterations=dilate_iterations)
        
        return dilated_image2

    def apply_watershed_algorithm(self, image, dilated_image2, maskSize, threshold_value):
        distance_transform = cv2.distanceTransform(dilated_image2, cv2.DIST_L2, maskSize)
        _, sure_fg = cv2.threshold(distance_transform, threshold_value * distance_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(dilated_image2, sure_fg)

        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(image, markers)
        
        return markers

    def process_markers_and_draw_bboxes(self, image, markers, model, min_size=40, max_iou=0.5, blue_threshold=0.0, red_yellow__threshold=1.0):
        height, width, _ = image.shape
        filtered_bboxes = []
        filtered_bboxes_normalized = []

        for i in range(2, np.max(markers) + 1):
            obj = {}
            points = np.where(markers == i)
            x, y, w, h = cv2.boundingRect(np.array([points[1], points[0]]).T)
            x2, y2 = x+w, y+h
            if model == "opencv":
                blue_percentage = self.calculate_blue_percentage(image, (x, y, x2, y2))
                red_yellow_percentage = self.calculate_red_yellow_percentage(image, (x, y, x2, y2))
            else:
                blue_percentage = 0.0
                red_yellow_percentage= 1.0
                
            is_blue = blue_percentage < blue_threshold
            is_red_yellow = red_yellow_percentage > red_yellow__threshold
            
            if w >= min_size and h >= min_size and self.is_valid_aspect_ratio((x,y,w,h)) and is_blue and is_red_yellow:
                current_bbox = (x, y,      w, h)
                obj['bbox'] = [y / height, x / width, (y + h) / height, (x + w) / width]

                should_add = True

                for existing_bbox in filtered_bboxes:
                    iou = self.bbox_iou(current_bbox, existing_bbox, width, height)
                    if iou > max_iou or self.is_bbox_inside(current_bbox, existing_bbox) or self.is_bbox_inside(existing_bbox, current_bbox):
                        should_add = False
                        break

                if should_add:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(image, f'RT:{h / w:.2f}', (x, y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)
                    cv2.putText(image, f'BL{blue_percentage:.2f}', (x, y + 40), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)
                    cv2.putText(image, f'RE{red_yellow_percentage:.2f}', (x, y + 60), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)
                    filtered_bboxes.append(current_bbox)
                    obj['class'] = 0.0
                    obj['score'] = 1.0

                    filtered_bboxes_normalized.append(obj)

        return filtered_bboxes_normalized

    def preprocess_image_and_detect_pedestrian(self, image, blue_threshold, red_yellow__threshold, model="opencv", min_size=40, max_iou=0.5, maskSize=5, threshold_value=0.5, erode_iterations=1, dilate_iterations=1):
        blurred_image = self.convert_to_grayscale_and_blur(image)
        thresh_image = self.threshold_image(blurred_image, threshold_value)
        dilated_image2 = self.apply_morphological_operations(thresh_image, erode_iterations, dilate_iterations)
        markers = self.apply_watershed_algorithm(image, dilated_image2, maskSize, threshold_value)
        filtered_bboxes_normalized = self.process_markers_and_draw_bboxes(image, markers, model, min_size, max_iou, blue_threshold, red_yellow__threshold)

        return dilated_image2, image, filtered_bboxes_normalized
