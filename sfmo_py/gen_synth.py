import numpy as np
from sfmo_py import config
from sfmo_py import utils

def get_extr(n_f, cam):
    """
    create the extrinsic parameters of the different cameras.
    """
    CamRt = np.zeros((n_f*3, 4))
    cam_t = np.array((0, 0, -cam['d']))
    azMx = cam['az']         # total azimuth angle
    elMx = cam['el']         # total elevation angle
    inc_t = cam['inc_t']      # step of the camera for each frame
    #  Initialization of projection matrix
    azmt = np.linspace(0, azMx, n_f)
    elv = np.linspace(0, elMx, n_f)
    for f in range(n_f):
        # Build the Projective Camera Matrix
        cam_R = utils.rot_euler2rot(np.array((azmt[f], elv[f], 0)) )
        CamRt[3*f:3*(f+1), :] = np.hstack((cam_R, np.reshape(cam_t,(3,1))))
        cam_t = cam_t + inc_t
    return CamRt


def get_orth(n_f, cam):
    """
    create the extrinsic parameters of the different cameras.
    """
    CamR = np.zeros((n_f*2, 3))
    azMx = cam['az']         # total azimuth angle
    elMx = cam['el']         # total elevation angle
    #  Initialization of projection matrix
    azmt = np.linspace(0, azMx, n_f)
    elv = np.linspace(0, elMx, n_f)
    for f in range(n_f):
        # Build the Projective Camera Matrix
        cam_R = utils.rot_euler2rot(np.array((azmt[f], elv[f], 0)) )
        CamR[2*f:2*f+2, :] = cam_R[:2,:] 
    return CamR


def create_cameras(n_f, cam, typ='persp'):
    if typ=='persp':
        P = get_extr(n_f, cam)
    else:
        P = get_orth(n_f, cam)
    return P


def create_objects(n_o, do_ell=False):
    if do_ell:
        quadrics = []
    else:
        quadrics = np.zeros((4*n_o,4))
    for o in range(n_o):
        center = np.random.normal(size=(3,1))*5
        angle = np.random.rand(3,1)*90 # in degree
        axis = np.random.rand(3,1)*10
        Q = utils.ellipsoid2quadric(center, angle, axis)
        quadrics[o * 4:(o + 1) * 4, :] = Q
    return quadrics


def reproj_objects(Ps,Qs, typ='persp'):
    n_o = Qs.shape[0]//4
    n_f = Ps.shape[0]//3
    C = np.zeros((3*n_f, 3*n_o))
    for f in range(n_f):
        for o in range(n_o):
            if typ == 'persp':
                P = np.vstack(( Ps[f*3:(f+1)*3-1, :], np.array((0, 0, 0, 1))))
            else:
                P = np.hstack(( Ps[f*2:f*2+2, :], np.zeros((2,1)) ))
                P = np.vstack(( P , np.array((0, 0, 0, 1))))
            Q = Qs[o*4:(o+1)*4]
            c = P.dot(Q).dot(P.transpose())
            C[3*f:3*(f+1), 3*o:3*(o+1)] = c / (-c[2,2])
    return C

def save_data(*args):
    if len(args) == 4:
        P, Q, C, filename = args
        pickle.dump((P, Q, C), open(filename, 'wb'))
    if len(args) == 2:
        data, filename = args
        pickle.dump(data, open(filename, 'wb'))


def get_conics_quadrics_cameras(typ='persp', n_f = config.n_f, cam = config.cam, n_o = config.n_o):
    P = create_cameras(n_f, cam, typ=typ)
    Q = create_objects(n_o)
    C = reproj_objects(P, Q, typ=typ)
    return (P, Q, C)


def get_matrix_inversion(d=3):
    m = np.random.rand(d, d)
    im = np.linalg.inv(m)
    return m.reshape(d*d), im.reshape(d*d)

def load_data(filename):
    return pickle.load(open(filename,'rb'))

if __name__ == "__main__":
    (P, Q, C) = get_conics_quadrics_cameras(typ='orth')
