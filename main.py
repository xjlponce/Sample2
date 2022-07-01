# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 14:09:59 2022

@author: Temporary1
"""
import torch
import cv2
from time import time
import PIL

# import pytesseract
import re
import numpy as np
import easyocr

import os
import glob

EASY_OCR = easyocr.Reader(['en'])
OCR_TH = 0.2


class Video(object):
    def __init__(self,model='best.pt'):

        
        self.video=cv2.VideoCapture(0)
        
        self.model = self.load_model(model)
        self.classes = self.model.names
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        
        
    def __del__(self):
        self.video.release()
        
    def detectx (self,frame, model):
        frame = [frame]
        print(f"[INFO] Detecting. . . ")
        results = model(frame)
        labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

        return labels, cordinates
    def load_model(self, model):

        model = torch.hub.load(r'D:\FINAL\Tutorial 7\yolov5-master', 'custom', path=r'D:\FINAL\Tutorial 7\yolov5-master\models\best.pt', source='local')
        return model
    
    def plot_boxes(self,results, frame,classes):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.9:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape) 
                
                coords = [x1,y1,x2,y2]

                plate_num = self.plate_easyocr(img = frame, coords= coords, reader= EASY_OCR, region_threshold= OCR_TH)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) 
                cv2.rectangle(frame, (x1, y1-20), (x2, y1), (0, 255,0), -1)
                cv2.putText(frame, f"{plate_num}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 2)
            
        return frame
    
    def plate_easyocr(self,img, coords,reader,region_threshold):
        # separate coordinates from box
        xmin, ymin, xmax, ymax = coords
        nplate = img[int(ymin):int(ymax), int(xmin):int(xmax)] ### cropping the number plate from the whole image

        #saving images in a directory

        #cv2.imread(nplate+".jpg")


        


        


        ocr_result = reader.readtext(nplate)
        text = self.filter_text(region=nplate, ocr_result=ocr_result, region_threshold= region_threshold)

        if len(text) ==1:
            text = text[0].upper()
        return text
    
    def filter_text(self,region,ocr_result,region_threshold):

        rectangle_size = region.shape[0]*region.shape[1]
        plate = [] 
        print(ocr_result)
        for result in ocr_result:
            length = np.sum(np.subtract(result[0][1], result[0][0]))
            height = np.sum(np.subtract(result[0][2], result[0][1]))
            
            if length*height / rectangle_size > region_threshold:
                plate.append(result[1])
                print(len(plate))
        return plate
    
    def get_frame(self):
        while True:
            model = torch.hub.load(r'D:\FINAL\Tutorial 7\yolov5-master', 'custom', path=r'D:\FINAL\Tutorial 7\yolov5-master\models\best.pt', source='local')
            classes = model.names
            ret, frame = self.video.read()
            frame = cv2.resize(frame, (416,416))
            start_time = time()
            
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            results = self.detectx(frame, model = model)
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            frame = self.plot_boxes(results, frame,classes = classes)
            end_time = time()
            fps = 1/np.round(end_time - start_time, 2)
            print(f"Frames Per Second : {fps}")
            
            ret,jpg=cv2.imencode('.jpg',frame)

            return jpg.tobytes()
            
        
        
        
        
    
    