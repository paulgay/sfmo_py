import json
import numpy as np
import utils
import cv2
import sys
tracking = json.load(open('tracking.json','r'))
colors = np.random.rand(6,3)*255

for (imgPath, dets) in tracking.items():
    image = cv2.imread(imgPath)
    print(imgPath)
    for (track_id, (x, y, w, h)) in dets:
        utils.draw_bounding_box(image, 'track'+str(track_id), colors[track_id], round(x), round(y), round(x+w), round(y+h))
    imname = utils.get_imname(imgPath)
    cv2.imwrite('tracking_'+imname+'.png', image)
