from attr import NOTHING
from matplotlib.font_manager import json_dump
import torch
import numpy as np
import cv2
from time import time
import json
import pandas as pd
from api_test import *
import csv

class ObjectDetection:
    """
    Class implements Yolo5 model to make inferences on a youtube video using Opencv2.
    """

    def __init__(self, capture_index, model_name):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
        self.capture_index = capture_index
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

    def get_video_capture(self):
        """
        Creates a new video streaming object to extract video frame by frame to make prediction on.
        :return: opencv2 video capture object, with lowest quality frame available for video.
        """
      
        return cv2.VideoCapture(self.capture_index)

    def load_model(self, model_name):
        """ 
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        model = torch.hub.load('yolov5', 'custom', path='Training\Run_UIObjfinalrun\\best.pt', source='local', force_reload=True)  # local repo

        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)          
        
        all_label = results.pandas().xyxy[0]["name"]
        all_cord = results.pandas().xyxy[0][["xmin","ymin","xmax","ymax"]]
        all_conf = results.pandas().xyxy[0]["confidence"]
        
        df= results.pandas().xyxy[0]
        df.to_csv("BA_Code\API_Annotations\dataframe.csv", encoding='utf-8', index=False)

        return all_label, all_cord, all_conf

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        all_labels, all_cord, all_conf = results
        n = len(all_labels)
        
        for i in range(n):

            cords = all_cord.iloc[i]
            conf = np.around(all_conf[i], decimals=2)
            label = all_labels[i]

            if conf >= 0.2:

                x1, y1, x2, y2 = int(cords[0]), int(cords[1]), int(cords[2]), int(cords[3])
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, label + " " + str(conf), (x1-50, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
                               
        return frame

    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        cap = self.get_video_capture()
        assert cap.isOpened()

        width  = int(cap.get(3))  # float `width`
        height = int(cap.get(4)) 
        print(width)
        print(height)
        cv2.namedWindow("YOLOv5 Detection")
        cv2.resizeWindow("YOLOv5 Detection", 1280, 720)
        createTrackbars(width, height)
                          
        
        while True:
          
            ret, frame = cap.read()
            assert ret
            
            y1 = cv2.getTrackbarPos("y1", "YOLOv5 Detection")
            y2 = cv2.getTrackbarPos("y2", "YOLOv5 Detection")
            x1 = cv2.getTrackbarPos("x1", "YOLOv5 Detection")
            x2 = cv2.getTrackbarPos("x2", "YOLOv5 Detection")
            
            if(y2 < y1):
                y2 = y1 + 1

            if(x2 < x1):
                x2 = x1 + 1
            
            cv2.normalize(frame, frame, cv2.getTrackbarPos("Alpha", "YOLOv5 Detection"), cv2.getTrackbarPos("Beta", "YOLOv5 Detection"), cv2.NORM_MINMAX)

            frame = frame[y1:y2, x1:x2]
            unedited_frame = frame
            takeimage(unedited_frame)

            start_time = time()
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            end_time = time()

            fps = 1/np.round(end_time - start_time, 2)
            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
            cv2.imshow('YOLOv5 Detection', frame)
            cv2.resizeWindow("YOLOv5 Detection", 1280, 720)

            #quit if q is pressed
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()
    
def takeimage(unedited_frame):
    #if s is pressesd send picture to robloflow via the api and increase image counter in the json file
            if cv2.waitKey(5) & 0xFF == ord('s'):
                
                with open("BA_Code\API_Images\image_counter.json", "r") as json_file:
                    my_dict = json.load(json_file)  
                    img_name = "opencv_frame_{}.png".format(my_dict[0]["img_counter"])
                    img_path = r"BA_Code\API_Images\\" + img_name
                    cv2.imwrite(img_path, unedited_frame)
                    print("{} written!".format(img_name))
                    print("IMAGE SAVED")
                    upload_image(img_path)   
                    #upload_annotation(upload_image(img_path))
                    

                with open("BA_Code\API_Annotations\dataframe.csv", 'r') as csv_file:
                    mycsv = csv.reader(csv_file)

                    data1 = []

                    for row in mycsv:
                        classname = row[6]
                        xmin = row[0]
                        ymin = row[1]
                        xmax = row[2]
                        ymax = row[3]
                        datarow = [img_name, classname, 832, 832, xmin,ymin,xmax,ymax]
                        data1.append(datarow)
                        #print(data1)
                    
                    del data1[0]
                    df = pd.DataFrame(data1, columns=["filename", "class", "width", "height", "xmin", "ymin", "xmax", "ymax"])
                    df.to_csv("BA_Code\API_Annotations\dataframe_anno.csv", encoding='utf-8', index=False)
                    csvtojson()


                with open("BA_Code\API_Images\image_counter.json", "w") as json_file:

                    json.dump([{"type": "integer", "img_counter": my_dict[0]["img_counter"] + 1}],json_file)

def createTrackbars(width, height):

    cv2.createTrackbar('Alpha', 'YOLOv5 Detection',
                    0, 255,
                    nothing) 
    cv2.createTrackbar('Beta', 'YOLOv5 Detection',
                    255, 255,
                    nothing) 
    cv2.createTrackbar('y1', 'YOLOv5 Detection',
                    70, height,
                    nothing) 
    cv2.createTrackbar('y2', 'YOLOv5 Detection',
                    600, height,
                    nothing)                 
    cv2.createTrackbar('x1', 'YOLOv5 Detection',
                    400, width,
                    nothing)        
    cv2.createTrackbar('x2', 'YOLOv5 Detection',
                    825, width,
                    nothing)  

def nothing(x):
    pass
# Create a new object and execute.
detector = ObjectDetection(capture_index=0, model_name='best.pt')
detector()