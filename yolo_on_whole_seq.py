import cv2
import argparse
import numpy as np
import sys, glob, os
import pickle

## SET THE PATHS ACCORDING TO YOUR CONFIGURATION
weights = #'/home/paul/programmation/darknet/yolov3.weights'  
config = #'/home/paul/programmation/darknet/cfg/yolov3.cfg'
class_file = #'/home/paul/programmation/darknet/data/coco.names'
inputFolder = 'image_sequence/'
out_file = 'yolo_detections.pc'


# function to get the output layer names 
# in the architecture
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def get_detection(image, net):
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392
   # create input blob 
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
    # set input blob for the network
    net.setInput(blob)
    # run inference through the network
    # and gather predictions from output layers
    outs = net.forward(get_output_layers(net))
    # initialization
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    # for each detetion from each output layer 
    # get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    boxes = [boxes[i[0]] for i in indices]
    confidences = [confidences[i[0]] for i in indices]
    class_ids = [class_ids[i[0]] for i in indices]
    return boxes, confidences, class_ids


# read class names from text file
classes = None
with open(class_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
# generate different colors for different classes 
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# read pre-trained model and config file
net = cv2.dnn.readNet(weights, config)
results = {}
for imgPath in glob.glob(inputFolder+"*.png"):
    print(imgPath)
    image = cv2.imread(imgPath)
    boxes, confidences, class_ids = get_detection(image, net)
    results[imgPath] = (boxes, confidences, [ classes[i] for i in class_ids])
    for (i,box) in enumerate(boxes):
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
    imname = os.path.basename(imgPath).replace('.png','')
pickle.dump(results, open(out_file,'wb'))
