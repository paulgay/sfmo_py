import json
import numpy as np
from sfmo_py import sfmo
from sfmo_py import utils
from sfmo_py import gen_synth


(P, Q, C) = gen_synth.get_conics_quadrics_cameras(typ='orth')
Cadjv_mat = utils.conics_to_vec(C, norm=True)
# Apply SFMO and reconstruct the quadrics.
Rec = sfmo.sfmo(Cadjv_mat)
# Convert them to ellipsoids, ie extract explicit centers, rotation and axis length parameters.
ells = utils.quadrics2ellipsoids(Rec)
# plot the results 
utils.plot_ellipsoids(ells)
