import json
import numpy as np
from sfmo_py import sfmo
from sfmo_py import utils


# Read tracking results 
tracking = json.load(open('tracking.json','r'))
n_o = 5 # number of objects
n_f = len(tracking) # number of frames 


# Detections are provided by YOLO detector, and we need ellipses. So the folling loop build an ellipse for each bounding box, and store the results in a matrix C of size (n_f*3, n_o*3)
C = np.zeros((n_f*3, n_o*3))
f = 0
for (imgPath,dets) in tracking.items():
    if len(dets) != n_o:
        continue # Sometimes, a detection might be missing and we don't deal with missing values.
    for (trackid, (x, y, w, h)) in dets:
        c = sfmo.bbx2ell((x, y, w, h))
        if np.abs(c.sum()) <= 0.1 or np.isnan(c).any():
            import pdb; pdb.set_trace()
        C[f*3:f*3+3, trackid*3:trackid*3+3] = c
    f += 1
C = C[:3*f]

# Vectoriser the conics stored in the C matrix: (n_f*3, n_o*3) -> (nf_*6, n_o)
Cadjv_mat = utils.conics_to_vec(C, norm=True)

# Apply SFMO and reconstruct the quadrics.
Rec = sfmo.sfmo(Cadjv_mat)
# Convert them to ellipsoids, ie extract explicit centers, rotation and axis length parameters.
ells = utils.quadrics2ellipsoids(Rec)
# plot the results 
utils.plot_ellipsoids(ells)
