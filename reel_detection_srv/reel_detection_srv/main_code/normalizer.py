import numpy as np
# import cv2
# import matplotlib.pyplot as plt


WIDTH = 640
HEIGHT = 480
def get_normalize_matrix(img):
    # POCOPHONE
    # K = np.array([[1207.096268, 0, 835.391219],
    #           [0, -1208.038718, 618.905228],
    #           [0, 0, 1]])
    # # 1663x1247

    # LAB
    K = np.array([[604.817626953125, 0, 317.8514404296875],
                  [0, 604.719482421875, 249.26316833496094],
                  [0, 0, 1]])
    
    [h,w] = img.shape[:2]
    rx = w/WIDTH
    ry = h/HEIGHT
    K[0,:] *= rx
    K[1,:] *= ry
    return K


def normalize_img_points(points, K):
    points = points.T
    points = np.vstack((points, np.ones(len(points[0]))))
    points = np.linalg.inv(K) @ points
    return points.T[:,:2]

def denormalize_img_points(points, K):
    points = points.T
    points = np.vstack((points, np.ones(len(points[0]))))
    points = K @ points
    return points.T[:,:2]
    
def denormalize_ellipse_coeffs(normed_coeffs, K):
    a,b,c,d,e,f = normed_coeffs
    A = np.array([[a, b/2, d/2],
                  [b/2, c, e/2],
                  [d/2, e/2, f]])
    
    # sx = 1/K[0,0]
    # sy = 1/K[1,1]
    # C =  sx**2 * sy**2 * K.T @ A @ K

    C = np.linalg.inv(K).T @ A @ np.linalg.inv(K)
    coeffs = np.array([C[0,0], 2*C[0,1], C[1,1], 2*C[0,2], 2*C[1,2], C[2,2]])
    return coeffs


if __name__ == "__main__":
    #  img = cv2.imread('reel_data/reel1.png',cv2.IMREAD_GRAYSCALE)
    #  [h,w] = img.shape[:2]
     
    #  K = get_normalize_matrix(img)
    #  K = np.linalg.inv(K)
    #  tl = K @ np.array([0,0,1]).T
    #  tr = K @ np.array([w,0,1]).T
    #  bl = K @ np.array([0,h,1]).T
    #  br = K @ np.array([w,h,1]).T
    #  points = np.vstack((tl,tr,bl,br)).T
     
    #  plt.plot(points[0,:], points[1,:],'bx')
    #  plt.show()
    pass