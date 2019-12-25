import numpy as np
import pickle
from munkres import Munkres
import json

def compare(b1, b2):
    """
    b1 = (x, y, w, h)
    b2 = (x, y, w, h)
    """
    (x1, y1, x2, y2) = (b1[0], b1[1], b1[0] + b1[2], b1[1] + b1[3])
    (x12, y12, x22, y22) = (b2[0], b2[1], b2[0] + b2[2], b2[1] + b2[3])
    inter = max(0, min(x2, x22) - max(x1, x12) ) * max(0, min(y2, y22) - max(y1, y12) )
    union = (x2 - x1) * (y2 - y1) + (x22 - x12) * (y22 - y12)
    return inter/union

def get_m(bb1, bb2):
    m = np.zeros((len(bb1), len(bb2)))
    for i, b1 in enumerate(bb1):
        for j, b2 in enumerate(bb2):
            m[i,j] = -compare(b1, b2)
    return m

def get_association(bb1, bb2):
    mat = get_m(bb1, bb2)
    m = Munkres()
    indexes = m.compute(mat.transpose())
    indexes = [(j, i) for (i,j) in indexes ]
    return indexes


n_o = 5

your_yolo_detection_file = "yolo_detections.pc"
results = pickle.load(open(your_yolo_detection_file,'rb'))
sorted_imgs = sorted(results.keys())
(bbox, conf, idclass) = results[sorted_imgs[0]]
tracks = [  [ box ]  for  (i, box) in enumerate(bbox)  if idclass[i]==39 ]
imgPaths = []
tracking = {}
for imgPath  in sorted_imgs[1:]:
    (bbox, conf, idclass) =  results[imgPath]
    bottles = [ box  for (i,box) in enumerate(bbox)  if idclass[i]==39 ]
    if len(bottles) > n_o:
        print("too many bottles",imgPath)
        continue
    last_bottles = [t[-1] for t in tracks]
    indices = get_association(last_bottles, bottles)
    tracking[imgPath] = []
    for i,j in indices:
        tracks[i].append(bottles[j])
        (x1, y1, w, h) = bottles[j]
        tracking[imgPath].append((i, (x1, y1, w, h)))

json.dump(tracking, open("tracking.json","w"))
