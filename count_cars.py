import cv2
from ultralytics import YOLO
import numpy as np
from tqdm import tqdm
import argparse
from utils import compute_iou

def count_cars(model, video, output_path):
    dict_classes = model.model.names
    vehicles = dict.fromkeys(class_IDS, 0)
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = video.get(cv2.CAP_PROP_FPS)

    cy_line_in = height - 120

    line_offset = 10
    frame_offset = (width//5, 30)

    count = 0

    vehicles = dict.fromkeys(class_IDS, 0)
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = video.get(cv2.CAP_PROP_FPS)

    cy_line_in = height - 120

    line_offset = 10
    frame_offset = (width//5, 30)

    count = 0

    output_video = cv2.VideoWriter(f'{output_path}',cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    prev_boxes = []
    for i in tqdm(range(int(video.get(cv2.CAP_PROP_FRAME_COUNT)))):
        
        _, frame = video.read()
        y_hat = model.predict(frame, conf = 0.7, classes = class_IDS, device = 0, verbose = False)

        boxes   = y_hat[0].boxes.xyxy.cpu().numpy()
        confs    = y_hat[0].boxes.conf.cpu().numpy()
        classes = y_hat[0].boxes.cls.cpu().numpy() 

        cv2.line(frame, (0, cy_line_in), (width, cy_line_in), (255,0,0),1)
        
        curr_boxes = []
        for box, conf, cl, in zip(boxes, confs, classes):
            xmin, ymin, xmax, ymax = box
            center_x, center_y = int((xmax+xmin)/2), int((ymax+ ymin)/2)
            
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255,0,0), 1) # box
            cv2.circle(frame, (center_x,center_y), 1,(255,0,0),-1)
            
            cv2.putText(img=frame, text=str(dict_classes[cl])+' - '+str(np.round(conf,2)),
                        org= (int(xmin),int(ymin)-10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.7, color=(255, 0, 0),thickness=1)

            if ((center_y < (cy_line_in + line_offset)) and (center_y > (cy_line_in - line_offset))):
                is_same_car = False
                for prev_box in prev_boxes:
                    iou = compute_iou(prev_box, box)
                    if iou > 0.5: 
                        is_same_car = True
                        break

                if not is_same_car:
                    count += 1
                    vehicles[cl] += 1
                curr_boxes.append(box)

        prev_boxes = curr_boxes

        cv2.putText(img=frame, text=f'Cars: {count}', 
                                org= (width-frame_offset[0], frame_offset[1]), fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
                                fontScale=1, color=(255, 0, 0),thickness=1)       

        output_video.write(frame)
        
    output_video.release()
    video.release()
    subprocess.run(
        ["ffmpeg",  "-i", output_path,"-crf","18","-preset","veryfast","-hide_banner","-loglevel","error","-vcodec","libx264",output_path])

        
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-video', type=str, required=True,
                help='The absolute/relative path to the video')

    # parse arguments
    args = vars(parser.parse_args())
    input_video_path = args.in_video
    output_video_path = input_video_path[:-4] + "_out.mp4"

    video = cv2.VideoCapture(input_video_path)
    class_IDS = [2, 5, 7]
    model = YOLO('yolov8x.pt')

    count_cars(model, video, output_video_path)
