import numpy as np


WIDTH = 640
HEIGHT = 480

def get_intrinsic_matrix(img):
    # POCOPHONE
    # K = np.array([[1207.096268, 0, 835.391219],
    #           [0, -1208.038718, 618.905228],
    #           [0, 0, 1]])
    # # 1663x1247
    
    # galaxy s23 3000x4000
    K_s23 = np.array([[2.678166215927513349e+03, 0.000000000000000000e+00, 1.999625919480884704e+03],
                      [0.000000000000000000e+00, 2.679508991653754947e+03, 1.453526119151558760e+03],
                      [0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]])
    # LAB
    K_480p = np.array([[604.817626953125, 0, 317.8514404296875],
                  [0, 604.719482421875, 249.26316833496094],
                  [0, 0, 1]])
    K_720p = np.array([[907.2265014648438, 0, 636.7771606445312],
                       [0, 907.0792236328125, 373.8947448730469],
                       [0, 0, 1]])
    
    if len(img) > 0 and img.shape[0] == 720:
        return K_720p

    # if len(img) > 0 and img.shape[0] == 3000:
    #     return K_s23
    # if len(img) > 0 and img.shape[0] == 960:
    #     [h,w] = img.shape[:2]
    #     rx = w/4000
    #     ry = h/3000
    #     K_s23[0,:] *= rx
    #     K_s23[1,:] *= ry
    #     return K_s23
    
    # if len(img) > 0:
    #     [h,w] = img.shape[:2]
    #     rx = w/WIDTH
    #     ry = h/HEIGHT
    #     K[0,:] *= rx
    #     K[1,:] *= ry
    
    return K_480p


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

def normalize_orient_points(orient_points, K):
    inv_K = np.linalg.inv(K)
    # orient_points = orient_points.T
    points = orient_points[:, :2]
    # gradients = orient_points[:, 2:]
    nx = orient_points[:, 2]
    ny = orient_points[:, 3]
    points = np.vstack((points.T, np.ones(len(points))))
    points = inv_K @ points
    points = points[:2,:]
    scale_x = inv_K[0,0]
    scale_y = inv_K[1,1]
    # alpha = np.sqrt(1 / ((nx*scale_x)**2 + (ny*scale_y)**2))
    # print(alpha)
    nx_hat = scale_x * nx
    ny_hat = scale_y * ny
    norms = np.sqrt(nx_hat**2+ny_hat**2)
    nx_hat = nx_hat / norms
    ny_hat = ny_hat / norms
    points = np.vstack((points, nx_hat))
    points = np.vstack((points, ny_hat))
        
    return points.T

def denormalize_orient_points(orient_points, K):
    points = orient_points[:, :2]
    nx = orient_points[:, 2]
    ny = orient_points[:, 3]
    points = np.vstack((points.T, np.ones(len(points))))
    points = K @ points
    points = points[:2,:]
    scale_x = K[0,0]
    scale_y = K[1,1]
    nx = scale_x * nx
    ny = scale_y * ny
    norms = np.sqrt(nx**2+ny**2)
    nx = nx / norms
    ny = ny / norms
    points = np.vstack((points, nx))
    points = np.vstack((points, ny))
    
    return points.T


def normalize_ellipse_coeffs(coeffs, K):
    a,b,c,d,e,f = coeffs
    A = np.array([[a, b/2, d/2],
                  [b/2, c, e/2],
                  [d/2, e/2, f]])
    
    # sx = 1/K[0,0]
    # sy = 1/K[1,1]
    # C =  sx**2 * sy**2 * K.T @ A @ K

    C = K.T @ A @ K
    coeffs = np.array([C[0,0], 2*C[0,1], C[1,1], 2*C[0,2], 2*C[1,2], C[2,2]])
    return coeffs

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
    print(C)
    return coeffs


if __name__ == "__main__":
    #  img = cv2.imread('reel_data/reel1.png',cv2.IMREAD_GRAYSCALE)
    #  [h,w] = img.shape[:2]
     
    #  K = get_intrinsic_matrix()
    #  K = np.linalg.inv(K)
    #  tl = K @ np.array([0,0,1]).T
    #  tr = K @ np.array([w,0,1]).T
    #  bl = K @ np.array([0,h,1]).T
    #  br = K @ np.array([w,h,1]).T
    #  points = np.vstack((tl,tr,bl,br)).T
     
    #  plt.plot(points[0,:], points[1,:],'bx')
    #  plt.show()
    pass