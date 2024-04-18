import numpy as np
import numpy.linalg as la
from numpy.linalg import norm

from reel_detection_srv.main_code import transformation as tr

def generate_Alvarez_matrix(coeffs):

    # ax2+bxy+cy2+dx+ey+f
    a,c,b,d,e,f = coeffs
    # A = np.array([[a,b,d/2],[b,c/2,e/2],[d/2,e/2,f]])
    A = np.array([[a,c/2,d/2],[c/2,b,e/2],[d/2,e/2,f]])
    evals, evecs = la.eig(A)
    if np.sum(np.sign(evals[:])) < 0:
        A = -A
        evals, evecs = la.eig(A)
    
    sorted_idx = np.argsort(evals)[::-1]
    sorted_evals = evals[sorted_idx]
    sorted_evecs = evecs[:, sorted_idx]

    M_eigen_sqrt = np.array([[1/np.sqrt(sorted_evals[0]),0,0],[0,1/np.sqrt(sorted_evals[1]),0],[0,0,1/np.sqrt(-sorted_evals[2])]])

    O = sorted_evecs.T
    tH = O.T @ M_eigen_sqrt

    lambda1,lambda2,lambda3 = sorted_evals[:]

    alpha = 0
    beta = 0

    val = lambda3*(lambda1-lambda2) / (lambda2*(lambda1+lambda3))
    
    a1 = np.sqrt(abs(val))
    a2 = -a1

    beta = np.pi
    Ra = np.array([[np.cos(alpha), np.sin(alpha), 0], [-np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])
    Rb = np.array([[np.cos(beta), np.sin(beta), 0], [-np.sin(beta), np.cos(beta), 0], [0, 0, 1]])
    
    Ba1 = np.array([[np.sqrt(1+a1**2), 0, a1], [0, 1, 0], [a1, 0, np.sqrt(1+a1**2)]])
    Ba2 = np.array([[np.sqrt(1+a2**2), 0, a2], [0, 1, 0], [a2, 0, np.sqrt(1+a2**2)]])

    H1 = tH @ Ra @ Ba1 @ Rb
    H2 = tH @ Ra @ Ba2 @ Rb

    return H1, H2

def create_Alvarez_matrix2(coeffs, K):

    # ax2+bxy+cy2+dx+ey+f
    a,c,b,d,e,f = coeffs
    # A = np.array([[a,b,d/2],[b,c/2,e/2],[d/2,e/2,f]])
    A = np.array([[a,c/2,d/2],[c/2,b,e/2],[d/2,e/2,f]])
    evals, evecs = la.eig(A)
    if np.sum(np.sign(evals[:])) < 0:
        A = -A
        evals, evecs = la.eig(A)
    
    sorted_idx = np.argsort(evals)[::-1]
    sorted_evals = evals[sorted_idx]
    sorted_evecs = evecs[:, sorted_idx]

    M_eigen_sqrt = np.array([[1/np.sqrt(sorted_evals[0]),0,0],[0,1/np.sqrt(sorted_evals[1]),0],[0,0,1/np.sqrt(-sorted_evals[2])]])

    O = sorted_evecs.T
    tH = O.T @ M_eigen_sqrt
    # print('O\n',O)
    # print('A\n',M_eigen_sqrt)
    # print('H(a)\n', tH)
    lambda1,lambda2,lambda3 = sorted_evals[:]

    # alpha = np.pi
    alpha = 0
    beta = 0
    # 0 and pi
    val = lambda3*(lambda1-lambda2) / (lambda2*(lambda1+lambda3))

    
    a1 = np.sqrt(abs(val))
    a2 = -a1
    
    Ra = np.array([[np.cos(alpha), np.sin(alpha), 0], [-np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])
    Rb = np.array([[np.cos(beta), np.sin(beta), 0], [-np.sin(beta), np.cos(beta), 0], [0, 0, 1]])
    
    Ba1 = np.array([[np.sqrt(1+a1**2), 0, a1], [0, 1, 0], [a1, 0, np.sqrt(1+a1**2)]])
    Ba2 = np.array([[np.sqrt(1+a2**2), 0, a2], [0, 1, 0], [a2, 0, np.sqrt(1+a2**2)]])

    H1 = tH @ Ra @ Ba1 @ Rb
    H2 = tH @ Ra @ Ba2 @ Rb
    

 # find correct H
    # l1_x0 = H1 @ np.array([[0,1,1],[0,0.8,1],[0,0,1],[0,-1,1]]).T
    # l1_x0 = l1_x0.T
    # l1_x0 = l1_x0 / l1_x0[:,-1][:,np.newaxis]
    # l2_x0 = H2 @ np.array([[0,1,1],[0,0.8,1],[0,0,1],[0,-1,1]]).T
    # l2_x0 = l2_x0.T
    # l2_x0 = l2_x0 / l2_x0[:,-1][:,np.newaxis]
    
    l1_y0 = H1 @ np.array([[1,0,1],[0.8,0,1],[0,0,1],[-1,0,1]]).T
    l1_y0 = l1_y0.T
    l1_y0 = l1_y0 / l1_y0[:,-1][:,np.newaxis]
    l2_y0 = H2 @ np.array([[1,0,1],[0.8,0,1],[0,0,1],[-1,0,1]]).T
    l2_y0 = l2_y0.T
    l2_y0 = l2_y0 / l2_y0[:,-1][:,np.newaxis]
    
    dir1 = l1_y0[2]-l1_y0[3]
    dir2 = l2_y0[2]-l2_y0[3]
    # rate1 = np.linalg.norm(dir1)/np.linalg.norm([l1_y0[0]-l1_y0[3]])
    # rate2 = np.linalg.norm(dir2)/np.linalg.norm([l2_y0[0]-l2_y0[3]])
    
    diay = l1_y0[0]-l1_y0[3]
    # diay2 = l2_y0[0]-l2_y0[3]
    rate1 = abs(dir1[1]/diay[1])
    rate2 = abs(dir2[1]/diay[1])
    
    rad = np.arctan2(dir1[1], dir1[0])
    angle = np.rad2deg(rad)
    
    dim = 1
    
    # l1_x0 = K @ l1_x0.T
    # l1_x0 = l1_x0.T
    # l2_x0 = K @ l2_x0.T
    # l2_x0 = l2_x0.T
    
    l1_y0 = K @ l1_y0.T
    l1_y0 = l1_y0.T
    l2_y0 = K @ l2_y0.T
    l2_y0 = l2_y0.T
    
    # l1_x0 = l1_x0[:,:-1]
    # l2_x0 = l2_x0[:,:-1]
    l1_y0 = l1_y0[:,:-1]
    l2_y0 = l2_y0[:,:-1]
    
    
    if (rate1 <= rate2) == (dir1[dim] >= 0):
        print('a =', a1)
        return H1, l1_y0, H2, l2_y0, dir1
    print('a =', a2)
    return H2, l2_y0, H1, l1_y0, dir2

def debug_create_Alvarez_matrix2(coeffs):

    # ax2+bxy+cy2+dx+ey+f
    a,c,b,d,e,f = coeffs
    # A = np.array([[a,b,d/2],[b,c/2,e/2],[d/2,e/2,f]])
    A = np.array([[a,c/2,d/2],[c/2,b,e/2],[d/2,e/2,f]])
    evals, evecs = la.eig(A)
    print('evals\n',evals)
    if np.sum(np.sign(evals[:])) < 0:
        A = -A
        evals, evecs = la.eig(A)
    
    sorted_idx = np.argsort(evals)[::-1]
    sorted_evals = evals[sorted_idx]
    sorted_evecs = evecs[:, sorted_idx]

    M_eigen_sqrt = np.array([[1/np.sqrt(sorted_evals[0]),0,0],[0,1/np.sqrt(sorted_evals[1]),0],[0,0,1/np.sqrt(-sorted_evals[2])]])

    O = sorted_evecs.T
    tH = O.T @ M_eigen_sqrt
    # print('O\n',O)
    # print('A\n',M_eigen_sqrt)
    # print('H(a)\n', tH)
    lambda1,lambda2,lambda3 = sorted_evals[:]

    alpha = 0
    beta = 0
    # 0 and pi
    val = lambda3*(lambda1-lambda2) / (lambda2*(lambda1+lambda3))

    
    a1 = np.sqrt(abs(val))
    a2 = -a1
    
    Ra = np.array([[np.cos(alpha), np.sin(alpha), 0], [-np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])
    Rb = np.array([[np.cos(beta), np.sin(beta), 0], [-np.sin(beta), np.cos(beta), 0], [0, 0, 1]])
    
    Ba1 = np.array([[np.sqrt(1+a1**2), 0, a1], [0, 1, 0], [a1, 0, np.sqrt(1+a1**2)]])
    Ba2 = np.array([[np.sqrt(1+a2**2), 0, a2], [0, 1, 0], [a2, 0, np.sqrt(1+a2**2)]])

    H1 = tH @ Ra @ Ba1 @ Rb
    H2 = tH @ Ra @ Ba2 @ Rb
    

 # find correct H
    l1_x0 = H1 @ np.array([[0,1,1],[0,0.5,1],[0,0,1],[0,-1,1]]).T
    l1_x0 = l1_x0.T
    l1_x0_hom = l1_x0 / l1_x0[:,-1][:,np.newaxis]
    l2_x0 = H2 @ np.array([[0,1,1],[0,0.5,1],[0,0,1],[0,-1,1]]).T
    l2_x0 = l2_x0.T
    l2_x0_hom = l2_x0 / l2_x0[:,-1][:,np.newaxis]
    
    l1_y0 = H1 @ np.array([[1,0,1],[0.5,0,1],[0,0,1],[-1,0,1]]).T
    l1_y0 = l1_y0.T
    l1_y0_hom = l1_y0 / l1_y0[:,-1][:,np.newaxis]
    l2_y0 = H2 @ np.array([[1,0,1],[0.5,0,1],[0,0,1],[-1,0,1]]).T
    l2_y0 = l2_y0.T
    l2_y0_hom = l2_y0 / l2_y0[:,-1][:,np.newaxis]
    
    dir1 = l1_y0_hom[2]-l1_y0_hom[3]
    dir2 = l2_y0_hom[2]-l2_y0_hom[3]
    # rate1 = np.linalg.norm(dir1)/np.linalg.norm([l1_y0_hom[0]-l1_y0_hom[3]])
    # rate2 = np.linalg.norm(dir2)/np.linalg.norm([l2_y0_hom[0]-l2_y0_hom[3]])
    
    diay1 = l1_y0_hom[0]-l1_y0_hom[3]
    diay2 = l2_y0_hom[0]-l2_y0_hom[3]
    print('length:', norm(diay1), norm(diay2))
    rate1 = abs(dir1[1]/diay1[1]) # ratio [0,-1] / [1,-1]
    rate2 = abs(dir2[1]/diay2[1])
    
    rad = np.arctan2(dir1[1], dir1[0])
    angle = np.rad2deg(rad)
    
    dim = 1
    
    # l1_x0 = K @ l1_x0.T
    # l1_x0 = l1_x0.T
    # l2_x0 = K @ l2_x0.T
    # l2_x0 = l2_x0.T
    
    # l1_y0 = K @ l1_y0.T
    # l1_y0 = l1_y0.T
    # l2_y0 = K @ l2_y0.T
    # l2_y0 = l2_y0.T
    
    # l1_x0 = l1_x0[:,:-1]
    # l2_x0 = l2_x0[:,:-1]
    # l1_y0 = l1_y0[:,:-1]
    # l2_y0 = l2_y0[:,:-1]
    cross_r1 = ((l1_y0_hom[0]-l1_y0_hom[1])/(l1_y0_hom[0]-l1_y0_hom[3]))/((l1_y0_hom[2]-l1_y0_hom[1])/(l1_y0_hom[2]-l1_y0_hom[3]))
    cross_r2 = ((l2_y0_hom[0]-l2_y0_hom[1])/(l2_y0_hom[0]-l2_y0_hom[3]))/((l2_y0_hom[2]-l2_y0_hom[1])/(l2_y0_hom[2]-l2_y0_hom[3]))
    print('cross ratio:', cross_r1, cross_r2)
    
    if (rate1 < rate2) == (dir1[dim] >= 0):
        print('a =', a1)
        return H1, l1_y0, H2, l2_y0, dir1, l1_x0, l2_x0
    print('a =', a2)
    return H2, l2_y0, H1, l1_y0, dir2, l2_x0, l1_x0

    # center1 = H1 @ np.array([0,0,1]).T
    # center1 = center1/center1[-1]
    # center2 = H2 @ np.array([0,0,1]).T
    # center2 = center2/center2[-1]
    # norm1 = np.linalg.norm(center1[:-1])
    # norm2 = np.linalg.norm(center2[:-1])
    # if (norm1 > norm2):
    #     print('a =', a1, a2)
    #     print('norms:', norm1, norm2)
    #     return H1, l1_y0, H2, l2_y0, dir1, l1_x0, l2_x0
    # print('a =', a2, a1)
    # print('norms:', norm2, norm1)
    # return H2, l2_y0, H1, l1_y0, dir2, l2_x0, l1_x0

def calc_point_line_distance(point, line_param):
    p = point.reshape((len(point),))
    A = line_param[0].reshape((len(line_param[0]),))
    D = line_param[1].reshape((len(line_param[1]),))
    
    t = ((p-A).T @ D) / (D.T @ D)
    p_l = A + t*D
    d = np.linalg.norm(p-p_l)
    return d, p_l

def find_closest_point_to_lines(lines_params):
    """ Line equation: L = A + tD

    Args:
        lines_params (array): array include two np array: [A, D]

    Returns:
        p: closest point to lines
    """
    I = np.eye(3)
    S = np.zeros((3,3))
    C = np.zeros((3))
    
    for param in lines_params:
        ai = param[0]
        ni = (param[1]/np.linalg.norm(param[1])).reshape((3,1))
        S += (I - np.outer(ni, ni.T))
        C += (I - np.outer(ni, ni.T)) @ ai
    
    Sp = np.linalg.pinv(S)
    p = Sp @ C
    
    return p

# def get_line_equation(K, point2d):
#     p_camera = np.linalg.inv(K) @ point2d
#     A = np.array([0,0,0])
#     D = p_camera/np.linalg.norm(p_camera)
#     return A, D
def get_two_line_distance(lines_params):
    A1, D1 = lines_params[0]
    A2, D2 = lines_params[1]
    d = 0
    if np.linalg.norm(np.cross(D1, D2)) == 0:
        B = D1
        d = abs(np.cross(B, A1-A2)) / np.linalg.norm(B)
    else:
        B = np.cross(D1, D2)
        d = abs(np.dot(B, A1-A2)) / np.linalg.norm(B)
    return d

def get_world_line_params(point_2d, K, rots, trans):
    T_cw = tr.get_w2c_transform_matrix(rots, trans)
    # point_2d = np.append(point_2d, 1.)
    point_cam = np.linalg.inv(K) @ point_2d
    point_cam = np.append(point_cam, 1.)
    point_w = T_cw @ point_cam
    point_w = point_w[:-1]
    focal = np.array([0,0,0,1])
    focal_w = T_cw @ focal
    focal_w = focal_w[:-1]
    D = point_w - focal_w
    D = D/np.linalg.norm(D)
    return focal_w, D

def find_ellipse_center(coeffs, is_acute = True):
    
    H1, H2 = generate_Alvarez_matrix(coeffs)
    
    unit_center = np.array([0,0,1])
    
    center1 = H1 @ unit_center
    center2 = H2 @ unit_center
    
    center1 = center1 / center1[-1]
    center2 = center2 / center2[-1]
    
    if (is_acute == (center1[1] < center2[1])):
        return center1
    return center2

if __name__ == "__main__":
    # bin_length = 3
    # for i in range(8):
    #    bin_list =  format(i, f'0{bin_length}b')
    #    print(bin_list[0])
    a = np.array([[1,2,3],[4,5,6]])
    print(a.tolist())
    