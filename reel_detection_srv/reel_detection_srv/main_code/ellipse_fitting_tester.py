import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import os
import glob

from reel_detection_srv.main_code import image_processor as improc
from reel_detection_srv.main_code import normalizer
from reel_detection_srv.main_code import ellipse_fitting as ef
from reel_detection_srv.main_code.my_utils import *
from reel_detection_srv.main_code import center_estimation as ce
    


# def debug_plot_3d_lines(lines, points):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
    
#     colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
#     for i in range(len(lines)):
#         line = rdm.generate_line_points(lines[i])
#         ax.plot(line[0], line[1], line[2], label='3D Line', color=colors[i])

#     for i in range(len(points)):
#         xp,yp,zp = points[i]
#         ax.scatter(xp, yp, zp, c=colors[i], marker='o')
        
#     # Set labels for the axes
#     ax.set_title('Reel center estimation in 3D')
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     # ax.grid(True)
#     ax.set_box_aspect([1, 1, 1])

#     # Display the plot
#     plt.show()

def get_ellipse_pts(params, npts=100, tmin=0, tmax=2*np.pi):
    """
    Return npts points on the ellipse described by the params = x0, y0, ap,
    bp, e, phi for values of the parametric variable t between tmin and tmax.

    """

    x0, y0, ap, bp, e, phi = params
    # A grid of the parametric variable, t.
    t = np.linspace(tmin, tmax, npts)
    x = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
    y = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
    return np.array([x, y])

def draw_ellipse(image, params, color = (255,0,0), thickness = 2):
    x0, y0, ap, bp, e, phi = params
    cv2.ellipse(image, (int(x0),int(y0)), (int(ap),int(bp)), phi*180/np.pi, 0, 360, color, thickness)

def get_pos_data(file_path):
    lines = []
    ret = []
    with open(file_path) as file:
        lines = file.readlines()
    for line in lines:
        data = line.split('[')
        for d in data:
            if len(d) > 0:
                d = d[:d.index(']')]
                d = d.split(' ')
                d = np.array([element for element in d if element])
                ret.append(d.astype(float))
    return ret

def points_to_orient_points(points, orientation):
    orient_mat = np.empty((0,2))
    for i in range(len(points)):
        x,y = points[i,:]
        # rad = orientation[y,x] + np.pi/2
        rad = orientation[y,x]
        nx = np.cos(rad)
        ny = np.sin(rad)
        orient_mat = np.vstack((orient_mat, np.array([nx,ny])))
    orient_points = np.hstack((points, orient_mat))
    
    return orient_points

def orient_ellipse_fitting(img, show_contour = False, show_result = False):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    orientation = improc.get_img_orient(img, ksize=5)
    [h,w] = img.shape[:2]
    
    result_img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    contours = improc.get_contours(img)
    # contours = improc.get_shattered_contours(img)
    contour_img = result_img.copy()
    sample_step_size = int((np.ceil(np.log(h*w)))/2)+2
    K = normalizer.get_intrinsic_matrix(img)
    
    normed_inlier_threshold = 0.0022
    inlier_threshold = 2
    confidence = 0.98
    max_failed_num = 1000 # max consecutive fail times
    max_eccentricity = 0.95
    max_iteration = 2000
    least_inliers = 0 # minimum inlier amount to be considered a candidate model
    min_inlier_rate = 0.9
    min_sample_dist = 2
    
    final_points = np.empty((0,4))
    
    for i_c in range(len(contours)):
        points = contours[i_c].squeeze(axis=1)
        points = points[::sample_step_size,:]
        # points = points[::2,:]
        
        orient_points = points_to_orient_points(points, orientation)
        coeffs = ef.orient_ellipse_fit(orient_points)
        # coeffs = ef.ellipse_fit(points)
        ellipse_params = ef.cart_to_pol(coeffs)
        eccentricity = ellipse_params[-2]

        if eccentricity > max_eccentricity:
            continue
        if not ef.check_orient_diffs(coeffs, orient_points, np.pi/18):
            continue
        
        inlier_indices = ef.get_inliers(orient_points, coeffs, inlier_threshold)
        inlier_rate = float(len(inlier_indices)) / len(orient_points)
        if inlier_rate < min_inlier_rate:
            continue
        
        improc.draw_orient_points(contour_img, orient_points)
        cv2.drawContours(contour_img, [contours[i_c]], -1, get_color(i_c), 2)
        draw_ellipse(contour_img, ef.cart_to_pol(coeffs))
        
        # orient_points = orient_points[::2,:]
        
        final_points = np.vstack((final_points, orient_points))
    
    if show_contour:  
        cv2.imshow(f, contour_img)
        cv2.waitKey()

# ellipse fitting 
    ransac_inlier_threshold = 1.5
    
    final_points = improc.remove_near_points(final_points, k=min(h,w)/64)
    improc.draw_points(result_img, final_points, 2, (255,0,0))
    # final_points = normalizer.normalize_orient_points(final_points, K)    
    best_coeffs, best_samples, best_inliers = ef.ransac_orient_ellipse_fit(final_points, ransac_inlier_threshold, confidence, max_iteration=max_iteration, max_failed_num=max_failed_num, max_eccentricity=max_eccentricity, min_sample_dist=min_sample_dist)

    if len(best_coeffs) > 0:
        # best_coeffs = normalizer.denormalize_ellipse_coeffs(best_coeffs, K)
        # best_samples = normalizer.denormalize_orient_points(best_samples, K)
        # best_inliers = normalizer.denormalize_orient_points(best_inliers, K)
        
        draw_ellipse(result_img, ef.cart_to_pol(best_coeffs), color=(0,255,255))
        improc.draw_orient_points(result_img, best_samples, line_length=10, line_thickness=2)
        improc.draw_points(result_img, best_inliers, 2, (255,255,0))
    
    #center estimation
        normed_coeffs = normalizer.normalize_ellipse_coeffs(best_coeffs, K)
        normed_center = ce.find_ellipse_center(normed_coeffs)
        center = K @ normed_center
        center_int = center.astype(int)
        # result_img = improc.draw_points(result_img, center, 4, (0,255,255))
        cv2.circle(result_img, (center_int[0], center_int[1]), 5, (0,255,0),-1)
    
    if show_result:
        cv2.imshow('result: '+f, result_img)
        cv2.waitKey()
    
    return best_coeffs, result_img, center        

def test_orient_ellipse_fitting(file_path, show_contour = False, show_result = False):
    print("file:", os.path.basename(file_path))
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    orient_ellipse_fitting(img, show_contour, show_result)

def direct_ellipse_fitting(img, show_contour = False, show_result = False):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    [h,w] = img.shape[:2]
    
    result_img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    contours = improc.get_contours(img)
    contour_img = result_img.copy()
    sample_step_size = int((np.ceil(np.log(h*w)))/2)+2
    K = normalizer.get_intrinsic_matrix(img)
    
    normed_inlier_threshold = 0.0022
    inlier_threshold = 1.5
    confidence = 0.95
    max_failed_num = 2000 # max consecutive fail times
    max_eccentricity = 0.9
    max_iteration = 4000
    least_inliers = 0 # minimum inlier amount to be considered a candidate model
    min_inlier_rate = 0.9
    min_sample_dist = 2
    
    final_points = np.empty((0,2))
    
    for i_c in range(len(contours)):
        points = contours[i_c].squeeze(axis=1)
        points = points[::sample_step_size,:]
        # points = points[::2,:]
        
        # coeffs = ef.orient_ellipse_fit(orient_points)
        coeffs = ef.ellipse_fit(points)
        ellipse_params = ef.cart_to_pol(coeffs)
        eccentricity = ellipse_params[-2]
        if eccentricity > max_eccentricity:
            continue
        
        inlier_indices = ef.get_inliers(points, coeffs, inlier_threshold)
        inlier_rate = float(len(inlier_indices)) / len(points)
        if inlier_rate < min_inlier_rate:
            continue
        
        cv2.drawContours(contour_img, [contours[i_c]], -1, get_color(i_c), 2)
        final_points = np.vstack((final_points, points[::2,:]))
        draw_ellipse(contour_img, ef.cart_to_pol(coeffs))
    
    if show_contour:
        cv2.imshow(f, contour_img)
        cv2.waitKey()
    
# ellipse fitting 
    ransac_inlier_threshold = 1
    final_points = improc.remove_near_points(final_points)
    improc.draw_points(result_img, final_points, 2, (255,0,0))
    # final_points = normalizer.normalize_orient_points(final_points, K)    
    best_coeffs, best_samples, best_inliers = ef.img_ransac_ellipse_fit(final_points, ransac_inlier_threshold, confidence, max_iteration=max_iteration, max_failed_num=max_failed_num, max_eccentricity=max_eccentricity, min_sample_dist=min_sample_dist)

    if len(best_coeffs) > 0:
        # best_coeffs = normalizer.denormalize_ellipse_coeffs(best_coeffs, K)
        # best_samples = normalizer.denormalize_orient_points(best_samples, K)
        # best_inliers = normalizer.denormalize_orient_points(best_inliers, K)
        
        draw_ellipse(result_img, ef.cart_to_pol(best_coeffs), color=(0,255,255))
        improc.draw_points(result_img, best_samples, 4, (0,0,255))
        improc.draw_points(result_img, best_inliers, 2, (255,255,0))
    
    #center estimation
        normed_coeffs = normalizer.normalize_ellipse_coeffs(best_coeffs, K)
        normed_center = ce.find_ellipse_center(normed_coeffs)
        center = K @ normed_center
        center_int = center.astype(int)
        # result_img = improc.draw_points(result_img, center, 4, (0,255,255))
        cv2.circle(result_img, (center_int[0], center_int[1]), 5, (0,255,0),-1)
        
    if show_result:
        cv2.imshow('result: '+f, result_img)
        cv2.waitKey()
    
    return best_coeffs, result_img, center        

def test_direct_ellipse_fitting(file_path, show_contour = False, show_result = False):
    print("file:", os.path.basename(file_path))
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    direct_ellipse_fitting(img, show_contour, show_result)
    
def test_fitting_in_folder(path, func):
    file_paths = get_file_paths(path, 'png', 'jpg', 'jpeg')
    for f in file_paths:
        func(f, False, False)

if __name__ == "__main__":

    # file_paths = get_file_paths('reels/20231201/reel_2_1280', 'png', 'jpg', 'jpeg')
    file_paths = get_file_paths('reels/reel_data', 'png', 'jpg', 'jpeg')
    # file_paths = get_file_paths('test_sample', 'png', 'jpg', 'jpeg')
    for f in file_paths:
        orient_ellipse_fitting(f, show_result=True)
        # test_ellipse_fitting(f, show_result=True)
        # time_func(test_ellipse_fitting,f)
        pass
    # time_func(test_fitting_in_folder, 'reels/reel_data', test_orient_ellipse_fitting)    
    # time_func(test_fitting_in_folder, 'reels/reel_data', test_ellipse_fitting)    
    pass

        # result_img, center1, center2, contour_img, debug = rdm.find_ellipse_center_2d_2(img)
        
        #depth method
        # center_3d, result_img, contour_img, L, debug = rdm.find_ellipse_center_3d_new(img, depth_img, rots, trans)
        # conims.append(contour_img)
        # ret_imgs.append(result_img)

        #stereo vision method
        # center_3d, result_imgs, lines_array, debug = rdm.find_ellipse_center_3d(imgs, rots_array, trans_array)
        # ret_imgs.append(result_imgs[0])
        # ret_imgs.append(result_imgs[1])
        
        # centers = np.vstack((centers, center_3d))
        # debugs.append([L,debug])
    
    # print('centers:\n', centers)
    # true_center[-1] -=200
    # # debug_plot_3d_lines([debugs[0][0]],[centers[0], true_center])
    # # print('debugs:\n', debugs)
    # error = centers[:]-true_center
    # error[:,-1] += 200
    # print('error:\n', error)
    # print('avg error:\n', np.mean(error,axis=0))
    
    # varx = np.var(centers[:,0])
    # vary = np.var(centers[:,1])
    # varz = np.var(centers[:,2])
    # print(f'var:{varx, vary, varz}')
    
    
    # plt.imshow(retim[:,:,::-1])
    # plt.show()
    # rdm.debug_plot_3d_lines([L], center_3d)

    