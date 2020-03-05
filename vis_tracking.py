import json
import numpy as np
import cv2
import sys
import os
tracking = json.load(open('tracking.json','r'))
colors = np.random.rand(6,3)*255

def draw_bounding_box(img, label, color, x, y, x_plus_w, y_plus_h):
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def get_imname(imgPath):
    return os.path.basename(imgPath).replace('.png','')

for (imgPath, dets) in tracking.items():
    image = cv2.imread(imgPath)
    print(imgPath)
    for (track_id, (x, y, w, h)) in dets:
        draw_bounding_box(image, 'track'+str(track_id), colors[track_id], round(x), round(y), round(x+w), round(y+h))
    imname = get_imname(imgPath)
    cv2.imwrite('tracking_'+imname+'.png', image)
