import scipy.linalg
import numpy as np
from sfmo_py import utils



def  bbx2ell( BB ):
    """
     Convert the rectangular bounding box BB into an ellipse
     plane q is tangent to the conic when:
     input: BB = [xmin ymin xmax ymax]
     output: C the 3x3 conic matrix
    
    0 = q S^-1 q^t
       = q M^-1 M S^-1 M^t M^t^-1 q^t
       = (q M^-1) (M S^-1 M^t) (q M^-1)^t
     
     transformed plane (q M^-1) should preserve incidence
     -> dual conic transformed by matrix M is:  (M S^-1 M^t)
    """
    BB = (BB[0], BB[1], BB[2] + BB[0], BB[3] + BB[1])
    a = abs(BB[2]-BB[0])/2
    b = abs(BB[3]-BB[1])/2
    
    x0  = np.array(((BB[0]+BB[2])/2, (BB[1]+BB[3])/2))
    Ccn = np.vstack(( np.hstack(( np.diag((1/a**2, 1/b**2)),np.zeros((2, 1)) )), np.array((0, 0, -1)) ))
    np.hstack(( np.eye(2,2),x0.reshape(2,1) ))
    P   = np.vstack(( np.hstack(( np.eye(2,2),x0.reshape(2,1))), np.array((0, 0, 1,)) ))
    C = P.dot(np.linalg.inv(Ccn)*np.linalg.det(Ccn)).dot(P.transpose());
    C = C / -C[2,2]
    return C


def fact_leq(r1, r2):
    v = np.array(( r1[0]*r2[0], r1[0]*r2[1]+r1[1]*r2[0], r1[0]*r2[2]+r1[2]*r2[0], r1[1]*r2[1], r1[1]*r2[2]+r1[2]*r2[1], r1[2]*r2[2] ))
    return v

def sfm_from_centers(C):
    """
    Apply a structure from motion pipeline to the centers of the conices 
    """
    # get number of objects and number of frames
    Nf, No = (C.shape[0]//6, C.shape[1])

    # Build a matrix which will contain the centers of the conics 
    matCentr = np.zeros((Nf*2,No))
    for f in range(Nf):
        matCentr[2*f,:] = -C[6*f+2,:]
        matCentr[2*f+1,:] = -C[6*f+4,:]

    # normalise this matrix by subtracting the mean of the 2D centres
    matCentr = matCentr - np.mean(matCentr,axis=1).reshape((Nf*2,1)).dot(np.ones((1,No)))

    # run SVD
    U, s, V = np.linalg.svd(matCentr)
    if U.shape[0] > V.shape[0]:
        D = np.vstack((np.diag(s), np.zeros((U.shape[0]-len(s),len(s)))  ))
    else:
        D = np.hstack((np.diag(s), np.zeros((len(s), V.shape[1]-len(s))) ))

    # Last multiplication
    D = D**.5
    M       = U.dot(D)
    S       = D.dot(V)

    # Get the motion and the structure
    M       = M[:,0:3]
    S       = S[0:3,:]
    return M, S

def fact_metric_constraint(Mhat, Shat):
    """
    Enforce orthographic constraint on the matrix Mhat and Shat
    """
    f  = Mhat.shape[0]//2
    M = np.zeros(Mhat.shape)
    S = np.zeros(Shat.shape)
    Q = np.zeros((3, 3))
    # Build the Q transformation matrix for Mhat. Q forces the orthogonality and the equality of the norm for every k 2*F by 3 matrix
    A = np.zeros((f*3+1,6)) # because fact_leq returns 6 length vectors
    for n in range(f):
        A[n] = fact_leq(Mhat[2*n], Mhat[2*n]) - fact_leq(Mhat[2*n+1], Mhat[2*n+1]) # equal norm condition 
        A[n+f] = fact_leq(Mhat[2*n], Mhat[2*n+1]) # orthogonality condition
        A[n+2*f] = fact_leq(Mhat[2*n+1], Mhat[2*n]) # orthogonality condition
    A[3*f,:] = fact_leq(Mhat[0], Mhat[0]) # fixed scale condition
    # Solve for v, the 6 unknown values of the symmetric value
    b = np.vstack(( np.zeros((3*f,1)), 1 ))
    v = np.linalg.lstsq(A,b, rcond=None) # A \         b   # Q*v = b
    v = v[0]
    # C is a symmetric 3x3 matrix such that C = G * G'
    C = np.zeros((3,3))
    C[0,0] = v[0]                  
    C[0,1] = v[1]                  
    C[0,2] = v[2]                  
    C[1,1] = v[3]
    C[1,2] = v[4]
    C[2,2] = v[5]
    C[1,0] = C[0,1]
    C[2,0] = C[0,2]
    C[2,1] = C[1,2]
    
    D = np.linalg.eigvals(C)
    if np.logical_or(np.iscomplex(D), D<0).any():
        print("Numerical instability, eigen values are complex or negative. WARNING, this part of the code has not been tested.")
        O1, lambd, O2 = svd(C)
        G = np.matmul(O1,np.diag(lambd**.5))
        num = np.zeros((2*f,G.shape[1]))
        den = np.zeros((2*f,G.shape[1]))
        for m in range(f):
            num[2*m:2*m+2] = np.matmul(Mhat[2*m:2*m+2], G)
            den[2*m:2*m+2] = np.linalg.pinv(np.matmul(Mhat[2*m:2*m+2],G).transpose())
        temp = np.linalg.lstsq(num,den)
        Q = np.matmul(G,scipy.linalg.sqrtm(temp))
    else:
        Q = np.linalg.cholesky(C).transpose()
    M = np.matmul(Mhat,Q)
    S = np.matmul(np.linalg.inv(Q), Shat)
    #error=sum(sum(abs(Mhat*Shat) - abs(M*S)))
    return M, S


def Pvectz( P ):
    """
    vectorize the P_i stacked in the input P = [P1, P2, P3 ]^T matrix
    each P_i is a 4 x 3 matrix
    """
    n_f = P.shape[0]//4
    P_vec = np.zeros((n_f, 12))
    for f in range(n_f):
        Pf = P[4*f:4*(f+1)].flatten()
        P_vec[f] = Pf
    return P_vec

def computeB(vecP):
    """
    COMPUTE B generates a matrix B [6 x 10] which correspond to the projection of the shape components from the quadrics to the conics
    """
    r = vecP[0:9]
    t = vecP[9:12]
    b1 = np.array((r[0]**2, 2 * r[0] * r[3], 2 * r[0] * r[6], 2 * r[0] * t[0], r[3]**2, 2 * r[3] * r[6], 2 * r[3] * t[0], r[6]**2, 2 * r[6] * t[0], t[0]**2))
    b2 = np.array((r[1] * r[0], r[1] * r[3] + r[4] * r[0], r[1] * r[6] + r[7] * r[0], t[1] * r[0] + r[1] * t[0], r[4] * r[3], r[4] * r[6] + r[7] * r[3], r[4] * t[0] + t[1] * r[3], r[7] * r[6], t[1] * r[6] + r[7] * t[0], t[1] * t[0] ))
    b3 = np.array((r[2] * r[0], r[2] * r[3] + r[5] * r[0], r[2] * r[6] + r[8] * r[0], t[2] * r[0] + r[2] * t[0], r[5] * r[3], r[5] * r[6] + r[8] * r[3], r[5] * t[0] + t[2] * r[3], r[8] * r[6], t[2] * r[6] + r[8] * t[0], t[2] * t[0]))
    b4 = np.array((r[1]**2, 2 * r[1] * r[4], 2 * r[1] * r[7], 2 * r[1] * t[1], r[4]**2, 2 * r[4] * r[7], 2 * r[4] * t[1], r[7]**2, 2 * r[7] * t[1], t[1]**2))
    b5 = np.array((r[2] * r[1], r[2] * r[4] + r[5] * r[1], r[2] * r[7] + r[8] * r[1], t[2] * r[1] + r[2] * t[1], r[5] * r[4], r[5] * r[7] + r[8] * r[4], r[5] * t[1] + t[2] * r[4], r[8] * r[7], t[2] * r[7] + r[8] * t[1], t[2] * t[1]))
    b6 = np.array((r[2]**2, 2 * r[2] * r[5], 2 * r[2] * r[8], 2 * r[2] * t[2], r[5]**2, 2 * r[5] * r[8], 2 * r[5] * t[2], r[8]**2, 2 * r[8] * t[2], t[2]**2))
    B = np.vstack((b1, b2, b3, b4, b5, b6))
    return B

def rebuild_Gred(R1):
    """
    Uses the orthographic matrix R1 to build the rest of the matrix which projects the quadrics into conics
    """
    Nf = R1.shape[0] // 2
    Gred = np.zeros((0,6))
    for f in range(Nf):
        Ptemp = R1[2*f:2*f+2,:]
        Ptemp = np.hstack((Ptemp, np.zeros((2,1)) ))
        Ptemp = np.vstack(( Ptemp, np.array((0,0,0,1)) ))
        vecP  = Pvectz(Ptemp.transpose());
        B     = computeB(vecP.reshape((12,)));
        Bnew  = B[np.ix_([0, 1, 3],[0, 1, 2, 4, 5, 7])]
        Gred  = np.vstack((Gred,Bnew))
    return Gred
    

def center_ellipses(C):
    """
    Centered the conics and p
    Input: 6 x Nf by No matrix, composed of 6 dim vectors corresponding to conics 
    Output: 3 x Nf by No matrix which contains the conic coefficient related to the shape once the conic has been centered. 
    """
    Nf, No = C.shape[0]//6, C.shape[1]
    Ccentered = np.zeros((Nf*3,No))
    centers = np.zeros((Nf*2,No))
    for f in range(Nf):
        for o in range(No):
            Cvec = C[6*f:6*f+6,o]
            T = np.eye(3,3)
            T[0,2] = Cvec[2]
            T[1,2] = Cvec[4]
            c_centered = utils.vec_conic(T.dot(utils.vec_to_con(Cvec)).dot(T.transpose()))
            Ccentered[3*f:3*f+3, o] = (c_centered[0],c_centered[1], c_centered[3] )
    return Ccentered

def recombine_ellipsoids(Qs, S):
    """
    Input: Qs: 3D centres 
    Add the 3D centres S to the elli
    """
    No = S.shape[1]
    Q_rec = np.zeros((4*No,4))
    for o in range(No):
        Q = np.array(((Qs[0,o], Qs[1,o], Qs[2,o], 0), (Qs[1,o], Qs[3,o], Qs[4,o], 0), (Qs[2,o], Qs[4,o], Qs[5,o], 0), (0, 0, 0, -1) )) #utils.vec_to_quad(Qs[:,o])
        T = np.eye(4,4)
        T[0:3,3] = S[:,o]
        Qr = T.dot(Q).dot(T.transpose())
        Q_rec[o*4:o*4+4] = Qr / (-Qr[3,3])
    return Q_rec


def sfmo(Cadjv_mat):
    """
    Cadfv_mat: matrix of vectorised conics [ c11, c22 ..] where cij is 6 dim vector which contains the jth object in the ith frame.
    """

    M, S = sfm_from_centers(Cadjv_mat)

    # Add additional orhogonality constraints
    M, S = fact_metric_constraint(M,S)
    # Build a rank 6 reduced matrix, eliminating rows and columns related to translation
    Gred = rebuild_Gred(M)

    # Remove center from ellipses (it is equivalent to center ellipsoids in the orthographic case)
    Ccenter = center_ellipses(Cadjv_mat)
    # Get the shape
    Quadrics_centered = np.linalg.lstsq(Gred,Ccenter)[0]
    # add the centers
    Rec  = recombine_ellipsoids(Quadrics_centered,S)
    return Rec

