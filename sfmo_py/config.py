import numpy as np
n_f = 25
n_o = 5
cam = {}
cam['f'] = 800                  # focal length in pixels
cam['w'] = 0
cam['h'] = 0
cam['d'] = 200       # camera distance from scene centre
cam['k'] = cam['h']/(2*60)         # Ortographic camera scale factor
# ~~~~~~~~ camera angle and position in the world frame ~~~~~~~~~~~~~~~
cam['Sz'] = 4
cam['angle'] = np.array((90, 180, 0))      # initial angle for the camera
cam['az'] = 10                   # angle variation 1 for each frame
cam['el'] = 10                   # angle variation 2 for each frame
cam['inc_t'] = np.array((3, 3, 0))      # step of the camera for each frame
K = np.array(( (cam['f'], 0, cam['w']/2), (0, cam['f'], cam['h']/2), (0, 0, 1) ))

kindOfCamera = 'ortho'
