# sfmo_py

Python implementation of the 2016 CVPR paper: "Structure from Motion with Objects (SfMO)". Given 2D object detections in multiple views, the method outputs affine camera matrices and 3D reconstruction as a set of 3D ellipsoides.

The original matlab code can be found [here](https://vgm.iit.it/code/structure-from-motion-with-objects).


![adding image](https://github.com/paulgay/sfmo_py/blob/master/images/sfmo.png)


A complete pipeline (object detection, tracking, and SfMO) is provided on a simple sequence. 

## installation

The sfmo module has few dependencies in itself. Some are needed for the tracking and the visualisation

```
pip install -r requirements.txt
```


## Running the SfMO pipeline. 

#### Real data

Main step: extract cameras and ellipsoids, and plot them: 

```
python do_sfmo_real_data.py
```
This script uses the file `tracking.json`, if you want to visualise what the tracking looks like, run: 

```
unzip bottle_seq.zip -d image_sequence
python vis_tracking.py
```


#### Synthetic data

Included is a module which generates synthetic cameras and ellipsoids and the corresponding ellipses projections. To run sfmo on synhtetic data, run: 

```
python do_sfmo_synth_data.py
```


#### Generate detection and tracking 

Download YOLO weights 


```
wget https://pjreddie.com/media/files/yolov3-openimages.weights 
unzip bottle_seq.zip -d image_sequence # if you did not did it before
python yolo_on_whole_seq.py # You need to set 

```

```

python tracking.py
```
It takes as input the file result.pc which contains the Yolo detections and outputs the file `tracking.json`.

This tracking is very naive, in the sense that the number of objects is assumed to be known, and it associates bounding boxes between consecutive frames by using only spatial proximity

## citation
Please cite this paper if you use the code:

```
@inproceedings{crocco2016structure,
  title={Structure from motion with objects},
  author={Crocco, Marco and Rubino, Cosimo and Del Bue, Alessio},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={4141--4149},
  year={2016}
}
```
