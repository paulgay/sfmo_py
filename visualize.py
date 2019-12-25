from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sfmo_py import utils
from sfmo_py import gen_synth


def get_min_max(ells, cameras):
    greatest_axes = max([max(ell['axes']) for ell in ells])
    max_ex, min_ex = max([ell['C'][0] for ell in ells]), min([ell['C'][0] for ell in ells])
    max_cx, min_cx = max([camera[0, :].max() for camera in cameras]), min([camera[0, :].min() for camera in cameras])
    max_x, min_x = max(max_cx, max_ex), min(min_cx, min_ex)

    max_ey, min_ey = max([ell['C'][1] for ell in ells]), min([ell['C'][1] for ell in ells])
    max_cy, min_cy = max([camera[1, :].max() for camera in cameras]), min([camera[1, :].min() for camera in cameras])
    max_y, min_y = max(max_cy, max_ey), min(min_cy, min_ey)

    max_ez, min_ez = max([ell['C'][2] for ell in ells]), min([ell['C'][2] for ell in ells])
    max_cz, min_cz = max([camera[2, :].max() for camera in cameras]), min([camera[2, :].min() for camera in cameras])
    max_z, min_z = max(max_cz, max_ez), min(min_cz, min_ez)
    return min_x - greatest_axes, min_y - greatest_axes, min_z - greatest_axes, max_x + greatest_axes, max_y + greatest_axes, max_z + greatest_axes

(P, Q, C) = gen_synth.get_conics_quadrics_cameras(typ='persp')

ells = utils.quadrics2ellipsoids(Q)
n_o = len(ells)
colors = np.random.rand(n_o, 3)


## plotting the ellipsoids and the cameras
fig = plt.figure(1)  # Square figure
ax = fig.add_subplot(111, projection='3d')
#plotting the ellipsoids
for (o, ell) in enumerate(ells):
    x, y, z = utils.get_ellipsoid_pc(ell)
    #Plot:
    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color=colors[o, :])

# getting the camera lines
cameras, lines = utils.get_cameras_pc(P[:3, :]) # only plotting the first 3 raws corresponding to the first camera
# plotting the cameras
for points in cameras:
    for i in range(lines.shape[1]):
        ax.plot([points[0, lines[0, i]], points[0, lines[1, i]]], [points[1, lines[0, i]], points[1, lines[1, i]]],
                zs=[points[2, lines[0, i]], points[2, lines[1, i]]], color=(0, 0, 1))
# Adjustment of the axes, so that they all have the same span:
min_x, min_y, min_z, max_x, max_y, max_z = get_min_max(ells, cameras)


getattr(ax, 'set_{}lim'.format('x'))((min_x, max_x))
getattr(ax, 'set_{}lim'.format('y'))((min_y, max_y))
getattr(ax, 'set_{}lim'.format('z'))((min_z, max_z))
plt.axis('equal')


# plotting one frame of the ellipses
utils.plot_ellipses(C, save_to=None, colors=None)

plt.show()

