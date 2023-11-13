import numpy as np
from numpy import cos, sin, pi
import re


    
def get_w2c_transform_matrix(rots, trans):
    T_tc = build_transform(get_rotation_matrix([45,0,0]), (-32.5, -105.287, 31.586))
    T_wt = build_transform(get_rotation_matrix(rots), trans)
    # pi_c = [0.,0.,0.]
    # pi_w = to_inv_homo(T_wt @ T_tc @ to_homo(pi_c))
    # T_wc = np.linalg.inv(T_wt @ T_tc)
    T_wc = T_wt @ T_tc
    return T_wc
    
    
def read_transform_data(path):
    with open(path, 'r') as file:
        content = file.read()
        
    pos_match = re.search(r'pos=array\(\[([^\]]+)\]', content)
    rot_match = re.search(r'rot=array\(\[([^\]]+)\]', content)
    
    if pos_match and rot_match:
        trans = pos_match.group(1)
        rots = rot_match.group(1)

        trans = np.array([float(num) for num in trans.split(',')], dtype=np.float64)
        rots = np.array([float(num) for num in rots.split(',')], dtype=np.float64)
    else:
        print("Arrays not found in the response.")
    return rots, trans
        
#build transformation matrix for robot
def build_transform(rotmat, translate):
    T = np.zeros((4,4))
    T[:3, :3] = rotmat
    T[:3,  3] = translate
    T[ 3,  3] = 1
    return T

def get_rotation_matrix(angles_deg):
    r, p, y = np.deg2rad(angles_deg) # roll, pitch, yaw
    R_z = np.array([[cos(y), -sin(y), 0],[sin(y), cos(y), 0], [0,0,1]])
    R_y = np.array([[cos(p), 0, sin(p)], [0,1,0], [-sin(p), 0, cos(p)]])
    R_x = np.array([[1,0,0], [0, cos(r), -sin(r)], [0, sin(r), cos(r)]])
    return R_z @ R_y @ R_x

#use homography on vector
def to_homo(v):
    v = np.array(v)
    res = np.ones(v.shape[0]+1)
    res[:v.shape[0]] = v
    return res

#use inverse homography on vector
def to_inv_homo(v):
    v = np.array(v)
    print(v)
    print(v[:-1])
    print(v[-1])
    return v[:-1] / v[-1]

if __name__ == '__main__':
    pass

