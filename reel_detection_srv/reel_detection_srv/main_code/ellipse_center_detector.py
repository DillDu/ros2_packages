import numpy as np
import numpy.linalg as la

def calc_tildeH(coeffs):
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
    # sorted_evecs = evecs[sorted_indices, :]

    𝚲 = np.array([[evals[0],0,0], [0,evals[1],0], [0,0,evals[2]]])
    sorted_𝚲 = np.array([[sorted_evals[0],0,0], [0,sorted_evals[1],0], [0,0,sorted_evals[2]]])

    # sorted_evals[:] = np.abs(evals[:])
    M_eigen_sqrt = np.array([[1/np.sqrt(sorted_evals[0]),0,0],[0,1/np.sqrt(sorted_evals[1]),0],[0,0,1/np.sqrt(-sorted_evals[2])]])

    O = sorted_evecs.T
    tH = O.T @ M_eigen_sqrt
    
    return tH

def create_transform_matrix2(coeffs, beta):

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

    alpha1 = np.pi
    alpha2 = np.pi/2

    # 0 and pi
    val1 = lambda3*(lambda1-lambda2) / (lambda2*(lambda1+lambda3))
    
    # pi/2 and 3/2pi
    val2 = lambda3*(lambda2-lambda1) / (lambda1*(lambda2+lambda3))

    
    a1 = np.sqrt(abs(val1))
    a2 = -a1

    a3 = np.sqrt(abs(val2))
    a4 = -a3

    Ra = np.array([[np.cos(alpha1), np.sin(alpha1), 0], [-np.sin(alpha1), np.cos(alpha1), 0], [0, 0, 1]])
    
    Rb = np.array([[np.cos(beta), np.sin(beta), 0], [-np.sin(beta), np.cos(beta), 0], [0, 0, 1]])
    
    Ba1 = np.array([[np.sqrt(1+a1**2), 0, a1], [0, 1, 0], [a1, 0, np.sqrt(1+a1**2)]])
    Ba2 = np.array([[np.sqrt(1+a2**2), 0, a2], [0, 1, 0], [a2, 0, np.sqrt(1+a2**2)]])

    H1 = tH @ Ra @ Ba1 @ Rb
    H2 = tH @ Ra @ Ba2 @ Rb

 # lower half of paper
    # center1 = H1 @ np.array([0,0,1]).T
    # center2 = H2 @ np.array([0,0,1]).T
    # center1 = center1/center1[-1]
    # center2 = center2/center2[-1]

    # s1 = (np.linalg.inv(tH) @ center1)[0] / -a1
    # s2 = (np.linalg.inv(tH) @ center1)[0] / -a2

    # m = (center2[1] - center1[1])/(center2[0]-center1[0])
    # b = center1[1]-m*center1[0]
    # line = np.array([m/b, -1/b,1]).T
    # l1 = Ba1.T @ Ra1.T @ tH.T @ line
    # l2 = Ba2.T @ Ra2.T @ tH.T @ line

    # beta1 = np.arctan2(-l1[1],l1[0])
    # beta2 = np.arctan2(-l2[1],l2[0])

    # print(line.T @ center1)
    # print(line.T @ center2)

    return H1, H2
    
def get_point_line_distance(point, line_param):
    p = point.reshape((len(point),))
    A = line_param[0].reshape((len(line_param[0]),))
    D = line_param[1].reshape((len(line_param[1]),))
    
    t = ((p-A).T @ D) / (D.T @ D)
    p_l = A + t*D
    d = np.linalg.norm(p-p_l)
    return d, p_l

def find_closest_point_to_lines(lines_params):
    I = np.eye(3)
    S = np.zeros((3,3))
    C = np.zeros((3,))
    
    for param in lines_params:
        ni = (param[1]/np.linalg.norm(param[1])).reshape((3,))
        ai = param[0].reshape((3,))
        S += (I - ni @ ni.T)
        C += (I - ni @ ni.T) @ ai
    
    Sp = np.linalg.pinv(S)
    p = Sp @ C
    
    return p