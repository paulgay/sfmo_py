# Structure from Motion with Objects (SfMO) in python 

<img src="https://github.com/paulgay/sfmo_py/blob/master/images/sfmo.png" alt="drawing" width="500" class="center"/>

Given 2D object detections in multiple views, the SfMO method outputs affine camera matrices and 3D reconstruction as a set of 3D ellipsoids. 

This repo is a python version of the original matlab code which can be found [here](https://vgm.iit.it/code/structure-from-motion-with-objets).

A complete pipeline (object detection, tracking, and SfMO) is provided on a simple sequence. 

## Installation

The SfMO module has few dependencies in itself. Some are needed for the tracking and the visualisation

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

Included is a module which generates synthetic cameras and ellipsoids and the corresponding ellipses projections. To run SfMO on synhtetic data, run: 

```
python do_sfmo_synth_data.py
```


#### Generate detection and tracking 

The detection has been done using [yolo](https://pjreddie.com/darknet/yolo/). Using the following script will generate a file 
```
unzip bottle_seq.zip -d image_sequence # Extract the images, if you did not did it before
python yolo_on_whole_seq.py # You need to set the filepath for yolo model inside the script.

```

```
python tracking.py
```
It takes as input the file result.pc which contains the Yolo detections and outputs the file `tracking.json`.

This tracking is very naive, in the sense that the number of objects is assumed to be known, and it associates bounding boxes between consecutive frames by using only spatial proximity and the hungarian algorithm.

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
