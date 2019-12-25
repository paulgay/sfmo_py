import math
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

conic_length = 6
quad_length = 10
ell_length = 6

def rot_euler2rot(euler):
    """
    rotation_euler2matrix
    :param euler: 3x1 vector containing the angles in degreee  
    :return: R the 3x3 rotation matrix
    """
    assert euler.shape[0] == 3
    R = np.zeros((euler.shape[0], euler.shape[0]))
    euler = euler * (math.pi / 180)

    cx = math.cos(euler[0])
    sx = math.sin(euler[0])
    cy = math.cos(euler[1])
    sy = math.sin(euler[1])
    cz = math.cos(euler[2])
    sz = math.sin(euler[2])

    R[0, 0] = cz * cy
    R[0, 1] = -sz * cy
    R[0, 2] = sy
    R[1, 0] = cz * sy * sx + sz * cx
    R[1, 1] = -sz * sy * sx + cz * cx
    R[1, 2] = -cy * sx
    R[2, 0] = -cz * sy * cx + sz * sx
    R[2, 1] = sz * sy * cx + cz * sx
    R[2, 2] = cy * cx
    return R


def rot_rot2euler(R):
    Rf = np.zeros((3, 3))
    Rf[:2, :] = R[:2, :]
    Rf[2,:] = np.cross(Rf[0,:], Rf[1,:])
    y = math.asin(Rf[0, 2])
    x = math.atan2(-Rf[1, 2] / math.cos(y), Rf[2,2] / math.cos(y))
    z = math.atan2(-Rf[0, 1] / math.cos(y), Rf[0, 0] / math.cos(y))
    eul = (360 / (2 * math.pi)) * np.array((x, y, z))
    return eul


def ellipsoid2quadric(center, angle, axis):
    center = np.reshape(center, (3, 1))
    angle = np.reshape(angle, (3, 1))
    axis = np.reshape(axis, (3, 1))
    B = np.square(np.diag((axis[:,0])))
    Q_0 = np.hstack(( np.vstack((B , np.zeros((1,3)) )), np.reshape(np.array((0,0,0,-1)),(4,1)) ))
    PEll_inv = np.vstack(( np.hstack((rot_euler2rot(angle),center )),  np.array((0,0,0,1)) ))
    return  PEll_inv.dot(Q_0).dot(PEll_inv.transpose())


def ell2quadric(ell):
    return ellipsoid2quadric(ell['C'], ell['euler'], ell['axes'])


def quadric2ellipsoid(Q):
    Q = Q / (-Q[3, 3])
    C = -Q[:3, 3]
    T = np.vstack((np.array((1, 0, 0, -C[0])), np.array((0, 1, 0, -C[1])), np.array((0, 0, 1, -C[2])), np.array((0, 0, 0, 1))))
    Qcent = T.dot(Q).dot(T.transpose())
    [D, V] = np.linalg.eig(Qcent[:3, :3])
    sort_ev = np.argsort(D)
    V = np.vstack((V[:, sort_ev[0]], V[:, sort_ev[1]], V[:, sort_ev[2]])).transpose()
    D.sort()
    if sum(D<0)>0:
        print("warning, eigen values are negatives")
        return None

    a = np.sqrt(D[0])
    b = np.sqrt(D[1])
    c = np.sqrt(D[2])
    euler = rot_rot2euler(V)
    ell = {'V': V, 'axes': np.array((a, b, c)), 'C': C, 'euler': euler}
    return ell


def quadrics2ellipsoids(Qs):
    ells = []
    n_o = Qs.shape[0]//4
    for o in range(n_o):
        ell = quadric2ellipsoid(Qs[o*4:(o+1)*4, :])
        ells.append(ell)
    return ells


def get_ellipsoid_pc(ell, size_side = 100):
    if not isinstance(ell, dict):
        #assuming it is a 4x4 quadric matrix
        ell = quadric2ellipsoid(ell)
    # Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, size_side)
    v = np.linspace(0, np.pi, size_side)
    # Cartesian coordinates that correspond to the spherical angles:
    # (this is the equation of an ellipsoid):
    x = ell['axes'][0] * np.outer(np.cos(u), np.sin(v))
    y = ell['axes'][1] * np.outer(np.sin(u), np.sin(v))
    z = ell['axes'][2] * np.outer(np.ones_like(u), np.cos(v))
    #rotate
    x, y, z = np.tensordot(ell['V'], np.vstack((x, y, z)).reshape((3, size_side, size_side)), axes=1)
    #translate to the ellipsoid centre
    x = x + ell['C'][0]
    y = y + ell['C'][1]
    z = z + ell['C'][2]
    return x, y, z


def get_camera_pc(P, size = 10):
    """
    :param P: a 3 x 4 extrinsic parameter matrix Rt
    :return: the points of the cameras,
             the "lines" variables, says which couple of points should be selected to draw the camera lines
    """
    assert P.shape[1] > 3 # camera should be 3 * 4 : [R r]
    # Points of the cube
    C_points1 = np.array(((1, 1, 1, 1, 0, 0, 0, 0), (1, 1, 0, 0, 1, 1, 0, 0), (1, 0, 1, 0, 1, 0, 1, 0)))

    # Points of the trapezoid
    C_points2 = np.array(((.75, 1, .75), (.25, 1, .75), (.25, 1, .25), (.75, 1, .25), (1, 2, 1), (1, 2, 0), (0, 2, 1),
                          (0, 2, 0))).transpose()
    points = np.hstack((C_points1, C_points2))
    q = np.array(((1, 0, 0), (0, 0, -1), (0, 1, 0)))
    points = q.dot(points)  # not sure why, maybe to align the z axis of the camera with a canonical wireframe
    lines = np.array([[ 0,  1,  3,  2,  6,  7,  5,  4,  0,  1,  5,  4,  6,  7,  3,  2,
                  0,  8,  9, 10, 11, 13, 15, 13, 12,  8, 12, 14,  9, 14, 15, 10,
                 15, 10, 11,  8,  0],
                [1,  3,  2,  6,  7,  5,  4,  0,  1,  5,  4,  6,  7,  3,  2,  0,
                  8,  9, 10, 11, 13, 15, 13, 12,  8, 12, 14,  9, 14, 15, 10, 15,
                 10, 11,  8,  0,  1]])
    points = size*P[:, :3].dot(points) + np.repeat(P[:, 3].reshape((3,1)), points.shape[1], axis=1)
    return points, lines


def vec_conic(C, norm=True):
    if norm:
        C = C/(-C[2, 2])
    vecC = np.zeros(conic_length)
    vecC[:3] = C[:, 0]
    vecC[3:5] = C[1:, 1]
    vecC[5] = C[2, 2]
    return vecC


def conics_to_vec(Cs, norm=True):
    """
    Arrange a matrix of 3f x 3o containing conic matrices into a matrix of 6F x o composed of 6 dimension vectorised conics
    """
    n_f = Cs.shape[0] // 3
    n_o = Cs.shape[1] // 3
    vec_cs = np.zeros((n_f*conic_length, n_o))
    for o in range(n_o):
        for f in range(n_f):
            C = Cs[f*3:(f+1)*3, o*3:(o+1)*3]
            vec_cs[f*conic_length: (f+1)*conic_length, o] = vec_conic(C, norm=norm)
    return vec_cs


def vec_conics(Cs):
    """
    arrange a matrix of 3f x 3o containing conic matrices into one vector
    """
    n_f = Cs.shape[0] // 3
    n_o = Cs.shape[1] // 3
    vec_cs = np.zeros(n_f*n_o*conic_length)
    for o in range(n_o):
        for f in range(n_f):
            C = Cs[f*3:(f+1)*3, o*3:(o+1)*3]
            vec_cs[o*n_f*conic_length + f*conic_length:o*n_f*conic_length + (f+1) * conic_length] = vec_conic(C, norm=norm)
    return vec_cs


def cam_to_vec(P):
    R = np.reshape(P[:, :3], 9)
    t = P[:, 3]
    return np.hstack((R, t))


def cams_to_vec(Ps):
    n_f = Ps.shape[0] // 3
    vec_ps = np.zeros(n_f*12)
    for f in range(n_f):
        vec_ps[f*12:(f+1)*12] = cam_to_vec(Ps[3*f:(f+1)*3, :])
    return vec_ps


def quad_to_vec(Q, norm=True):
    if norm:
        Q = Q/(-Q[3, 3])
    vecQ = np.zeros(quad_length)
    vecQ[:4] = Q[0, :]
    vecQ[4:7] = Q[1, 1:]
    vecQ[7:9] = Q[2, 2:]
    vecQ[9] = Q[3, 3]
    return vecQ


def quads_to_vecs(Qs, norm=True):
    n_o = Qs.shape[0] // 4
    vec_qs = np.zeros(quad_length*n_o)
    for o in range(n_o):
        vec_qs[quad_length*o:(o+1)*quad_length] = quad_to_vec(Qs[o*4:(o+1)*4], norm=norm)
    return vec_qs


def ell_to_vec(ell):
    v = np.zeros(6)
    v[:3] = ell['C']
    ax = ell['axes']
    np.random.shuffle(ax)
    v[3:6] = ax
    #v[6:] = ell['euler']
    return v


def ells_to_vecs(ells):
    n_o = len(ells)
    vec_ells = np.zeros(ell_length*n_o)
    for o in range(n_o):
        vec_ells[ell_length*o:(o+1)*ell_length] = ell_to_vec(ells[o])
    return vec_ells

def reproj_objects(Ps,Qs, cam_type): 
    n_o = Qs.shape[0]//4 
    n_f = Ps.shape[0]//3 
    C = np.zeros((3*n_f, 3*n_o)) 
    for f in range(n_f): 
        for o in range(n_o): 
            if cam_type == 'persp':
                P = np.vstack(( Ps[f*3:(f+1)*3-1, :], np.array((0, 0, 0, 1)))) 
            else: # orthographic
                P = Ps[2*f:2*f+2]
                P = np.vstack(( np.hstack((P,np.zeros((2,1)))), np.array((0, 0, 0, 1))  ))
            Q = Qs[o*4:(o+1)*4] 
            C[3*f:3*(f+1), 3*o:3*(o+1)] = P.dot(Q).dot(P.transpose())
    return C 

def vec_to_quad(vec, norm=True):
    Q = np.zeros((4, 4))
    Q[0, :] = vec[:4]
    Q[1, 1:] = vec[4:7]
    Q[2, 2:] = vec[7:9]
    Q[3, 3] = vec[9]
    Q = Q + Q.transpose() - np.diag(Q.diagonal())
    if norm:
        Q = Q / (-Q[3, 3])
    return Q


def vecs_to_quads(vecs):
    n_o = vecs.shape[0] // quad_length
    quads = np.zeros((n_o*4, 4))
    for o in range(n_o):
        vec = vecs[o*quad_length:(o+1)*quad_length]
        quads[o*4:(o+1)*4, :] = vec_to_quad(vec)
    return quads


def vec_to_ell(vec):
    Q = vec_to_quad(vec)
    return quadric2ellipsoid(Q)


def vec_to_cam(vec):
    R = vec[:9].reshape((3, 3))
    t = vec[9:12].reshape((3,1))
    return np.hstack((R, t))


def vecs_to_cams(vecs):
    n_f = vecs.shape[0] // 12
    P = np.zeros((3*n_f, 4))
    for f in range(n_f):
        P[f*3:(f+1)*3, :] = vec_to_cam(vecs[12*f:(f+1)*12])
    return P


def get_cameras_pc(P):
    n_f = P.shape[0] // 3
    cameras = []
    for f in range(n_f):
        p = P[3*f:3*(f+1),:]
        points, lines = get_camera_pc(p)
        cameras.append(points)
    return cameras, lines


def get_ellipse(C):
    cent = -C[2, :2]
    T = np.vstack((np.array((1, 0, -cent[0])), np.array((0, 1, -cent[1])), np.array((0, 0, 1))))
    Ccent = T.dot(C).dot(T.transpose())
    [D, V] = np.linalg.eig(Ccent[:2, :2])
    ax1 = np.sqrt(abs(D[0]))
    ax2 = np.sqrt(abs(D[1]))
    axes = np.array((ax1, ax2))
    R = -V # don't know why, to get results coherent with the matlab function. Maybe it align the axis of the ellipse with the reference frame.
    return cent, axes, R


def get_ellipses(C):
    n_f = C.shape[0] // 3
    n_o = C.shape[1] // 3
    ellipses = {}
    for f in range(n_f):
        ellipses[f] = []
        for o in range(n_o):
            cent, axes, R = get_ellipse(C[f*3:(f+1)*3, o*3:(o+1)*3])
            ang = np.cos(R[0, 0]) / np.pi * 180
            e = Ellipse(xy=cent, width=axes[0], height=axes[1], angle=ang)
            ellipses[f].append(e)
    return ellipses

def vec_to_con(v):
    C = np.zeros((3,3))
    C[:, 0] = v[:3]
    C[1:, 1] = v[3:5]
    C[2, 2] = v[5]
    C = C + C.transpose() - np.diag(C.diagonal())
    return C


def vecs_to_cs(vecs, n_f):
    n_o = vecs.shape[0] // (conic_length * n_f)
    C = np.zeros((3*n_f, 3*n_o))
    for f in range(n_f):
        for o in range(n_o):
            v = vecs[o*n_f*conic_length + f*conic_length:o*n_f*conic_length + (f+1) * conic_length]
            C[o*3:(o+1)*3, f*3:(f+1)*3] = vec_to_con(v)
    return C


def get_min_max_c(ells_f):
    greatest_axes = max([max(e.width, e.height) for e in ells_f])
    max_x, max_y = max([e.center[0] for e in ells_f]), max([e.center[1] for e in ells_f])
    min_x, min_y = min([e.center[0] for e in ells_f]), min([e.center[1] for e in ells_f])
    return min_x - greatest_axes, min_y - greatest_axes, max_x + greatest_axes, max_y + greatest_axes


def plot_ellipses(C, save_to=None, colors=None):
    #getting the ellipses
    ells = get_ellipses(C)
    n_o = len(ells[0])  # guessing the number of objects from the first frame
    #plotting the ellipses
    if colors is None:
        colors = np.random.rand(n_o, 3)
    for n_f, ells_f in ells.items():
        fig2 = plt.figure(2)
        ax2 = fig2.add_subplot(111, aspect='equal')
        for o, e in enumerate(ells_f):
            ax2.add_artist(e)
            e.set_clip_box(ax2.bbox)
            e.set_alpha(0.5)
            e.set_facecolor(color=colors[o, :])
            min_x, min_y, max_x, max_y = get_min_max_c(ells_f)
            #print(min_x, min_y, max_x, max_y)
            getattr(ax2, 'set_{}lim'.format('x'))((min_x, max_x))
            getattr(ax2, 'set_{}lim'.format('y'))((min_y, max_y))
        break # just print one frame, but I could do more
    if save_to ==None:
        plt.show()
    else:
        plt.savefig(save_to)


def plot_ellipsoids(ells):
    n_o = len(ells)
    colors = np.random.rand(n_o, 3)
    #fig = plt.figure(figsize=plt.figaspect(1))  # Square figure
    fig = plt.figure(1)  # Square figure
    ax = fig.add_subplot(111, projection='3d')
    #plotting the ellipsoids
    for (o, ell) in enumerate(ells):
        x, y, z = get_ellipsoid_pc(ell)
        #Plot:
        ax.plot_surface(x, y, z,  rstride=4, cstride=4, color=colors[o, :])
    plt.show()
    



