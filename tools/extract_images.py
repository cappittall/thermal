import cv2
import glob

videos = glob.glob('data/videos/*.*')
for video in videos:
    video_name = video.split('/')[-1].split('.')[0]
    cam1 = cv2.VideoCapture(video)
    counter = 0
    while True:
        counter += 1
        ret, frame = cam1.read()
        if not ret: break

        if counter % 5 == 0:
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imname = f'data/images/img{video_name}_{counter:03d}.jpg'
            cv2.imwrite(imname, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])  # Set JPEG quality to 95
            print('Video name: ', video_name, 'Image Name : ', imname)

            
    
    