import numpy as np
import cv2
# import matplotlib.pyplot as plt

from reel_detection_srv.main_code import ellipse_fitting as ef
from reel_detection_srv.main_code import ellipse_center_detector as ecd
from reel_detection_srv.main_code import image_processor as improc
from reel_detection_srv.main_code import normalizer
from reel_detection_srv.main_code import utils
from reel_detection_srv.main_code import transformation as tr

def generate_line_points(line_params, interval=[0,2000], resolution=10000):
    A, D = line_params
    t = np.linspace(interval[0], interval[1], 100)
    line = np.array([[A[0]+D[0]*t],
                        [A[1]+D[1]*t],
                        [A[2]+D[2]*t]]).reshape(3,len(t))
    return line

def generate_ellipse_pts(params, npts=100, tmin=0, tmax=2*np.pi):
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

def get_final_points(coeffs_array, point_array):
    areas = []
    for coeffs in coeffs_array:
        x0, y0, ap, bp, e, phi = ef.cart_to_pol(coeffs)
        areas.append(ap*bp)
    sorted_indices = sorted(range(len(areas)), key=lambda k: areas[k])
    sorted_indices = sorted_indices[::-1]
    sorted_indices = sorted_indices[:int(np.ceil(len(sorted_indices)/2))]
    
    final_points = np.empty((0,2))
    for i in sorted_indices:
        final_points = np.vstack((final_points, point_array[i]))
    return final_points

def find_ellipse_center_2d(img):
    #ransac params
    inlier_threshold = 0.0022
    confidence = 0.95
    max_failed_num = 20 # max consecutive fail times
    max_eccentricity = 0.7
    max_iteration = 100
    least_inliers = 0 # minimum inlier amount to be considered a candidate model
    min_inlier_rate = 0.7
    min_sample_dist = 0
    
    img = improc.preproc(img, output_gray=True)
    
    [h,w] = img.shape[:2]
    result_img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    contours = improc.get_contours(img)
    # contours = improc.simplify_contours(contours, 0.1)

    point_array = []
    contour_coeffs = []
    final_points = np.empty((0,2))
    contour_img = result_img.copy()
    sample_step_size = int((np.ceil(np.log(h*w)))/2)+2

    K = normalizer.get_intrinsic_matrix(img)
    
    for i in range(len(contours)):
        points = contours[i].squeeze(axis=1)
        points = points[::sample_step_size,:]
        points = points[::2,:]
        # final_points = np.vstack((final_points, points))
        cv2.drawContours(contour_img, [contours[i]], -1, utils.get_color(i,True), 2)
        
        normed_points = normalizer.normalize_img_points(points, K)
        coeffs, samples, inliers = ef.img_ransac_ellipse_fit(normed_points, inlier_threshold, confidence, max_iteration=max_iteration, max_failed_num=max_failed_num, max_eccentricity=max_eccentricity, min_sample_dist=min_sample_dist)

        if len(inliers) / len(points) >= min_inlier_rate:
            final_points = np.vstack((final_points, points))
            coeffs = normalizer.denormalize_ellipse_coeffs(coeffs, K)
            ellipse_points = generate_ellipse_pts(ef.cart_to_pol(coeffs), npts=100).T
            # contour_img = improc.draw_img_points(contour_img, points, 3, utils.get_color(i,True))
            contour_img = improc.draw_img_points(contour_img, ellipse_points, 3, (255,0,0))
            # contour_img = improc.draw_img_points(contour_img, inliers, 1, (0,255,0))
            contour_coeffs.append(coeffs)
            point_array.append(points)
            
    # cv2.imshow('contour',contour_img)

    final_points = get_final_points(contour_coeffs, point_array)
    #final fitting
    if len(final_points) == 0:
        # raise SystemExit
        return 0,0,0,0
    inlier_threshold = 0.002
    confidence = 0.95
    max_failed_num = 100 # max consecutive fail times
    max_eccentricity = 0.8
    max_iteration = 400
    least_inliers = 0 # minimum inlier amount to be considered a candidate model
    min_inlier_rate = 0.6
    min_sample_dist = 1
    
    result_img = improc.draw_img_points(result_img, final_points, 1, (255,255,0))
    # final_points = np.array(final_points)
    final_points = normalizer.normalize_img_points(final_points, K)
    normed_best_coeffs, samples, inliers = ef.img_ransac_ellipse_fit(final_points, inlier_threshold, confidence, max_failed_num=max_failed_num, max_eccentricity=max_eccentricity, least_inliers=least_inliers, min_sample_dist=min_sample_dist)
    
    if len(normed_best_coeffs) == 0:
        return 0,0,0,0
    
    best_coeffs = normalizer.denormalize_ellipse_coeffs(normed_best_coeffs, K)
    samples = normalizer.denormalize_img_points(samples, K)
    inliers = normalizer.denormalize_img_points(inliers, K)

    x0, y0, ap, bp, e, phi = ef.cart_to_pol(best_coeffs)
    ellipse_points = generate_ellipse_pts([x0, y0, ap, bp, e, phi], npts=500).T
    result_img = improc.draw_img_points(result_img, final_points, 3, (255, 0, 0))
    result_img = improc.draw_img_points(result_img, ellipse_points, 2, (0, 128, 255))
    result_img = improc.draw_img_points(result_img, samples, 4, (0,0,255))
    result_img = improc.draw_img_points(result_img, inliers, 1, (0,255,0))
    
    # H1,H2 = ecd.create_Alvarez_matrix(best_coeffs, beta)
    
    # center1 = H1 @ np.array([0,0,1]).T
    # center2 = H2 @ np.array([0,0,1]).T

    # center1 = center1/center1[-1]
    # center2 = center2/center2[-1]
    
    # center1 = K @ center1
    # center2 = K @ center2
    
    # result_img = cv2.circle(result_img, (int(center1[0]),int(center1[1])),5,(255,0,0),-1)
    # result_img = cv2.circle(result_img, (int(center2[0]),int(center2[1])),5,(0,255,0),-1)
    
    H, l_y0, H2, l2_y0, debug = ecd.create_Alvarez_matrix2(normed_best_coeffs, K)
    center = H @ np.array([0,0,1]).T
    center = center/center[-1]
    center2 = H2 @ np.array([0,0,1]).T
    center2 = center2/center2[-1]
    
# if two center too close, get mean
    # avg_center = []
    # center_dist = np.linalg.norm(center-center2)
    # debug = center_dist
    # if center_dist < 0.015:
    #     avg_center = np.mean([center, center2], axis=0)
    #     # avg_center = K @avg_center
    #     center = avg_center
    
    center = K @ center
    center2 = K @ center2
    
    # l_x0 = l_x0.astype(int)
    l_y0 = l_y0.astype(int)
    # result_img = cv2.line(result_img, l_x0[0], l_x0[-1], color=(0,255,255), thickness=2)
    # result_img = cv2.line(result_img, l_y0[0], l_y0[-1], color=(0,255,255), thickness=2)
    # result_img = cv2.line(result_img, l2_x0[0], l2_x0[-1], color=(255,255,0), thickness=2)
    # result_img = cv2.line(result_img, l2_y0[0], l2_y0[-1], color=(255,255,0), thickness=2)
    # result_img = improc.draw_img_points(result_img, l_x0,5,(255,0,0))
    result_img = improc.draw_img_points(result_img, l_y0,3,(255,0,0))
    # result_img = improc.draw_img_points(result_img, l2_x0,5,(0,0,255))
    result_img = improc.draw_img_points(result_img, l2_y0,3,(0,0,255))
    result_img = cv2.circle(result_img, (int(center[0]),int(center[1])),5,(255,0,0),-1)
    result_img = cv2.circle(result_img, (int(center2[0]),int(center2[1])),5,(0,0,255),-1)
    # if len(avg_center) > 0:
    #     result_img = cv2.circle(result_img, (int(avg_center[0]),int(avg_center[1])),7,(0,255,255),-1)

    # return result_img, center1, center2, contour_img
    return result_img, center, center2, contour_img, debug
    
def find_ellipse_center_3d(imgs, rots_array, trans_array):
    K = normalizer.get_intrinsic_matrix()
    imgs_num = len(imgs)
    centers_array = []
    lines_array = []
    result_imgs = []
    for i in range(imgs_num):
        img = imgs[i]
        rots = rots_array[i]
        trans = trans_array[i]
        
        result_img, center1, center2, contour_img, debug = find_ellipse_center_2d(img)
        # centers = [center1,center2]
        # centers_array.append(centers)
        
        L1 = ecd.get_world_line_params(center1, K, rots, trans)
        L2 = ecd.get_world_line_params(center2, K, rots, trans)
        lines = [L1, L2]
        lines_array.append(L1)
        result_imgs.append(result_img)
        
# generate line pairs
    # group_num = 2**imgs_num
    # line_groups = []
    # for i in range(group_num):
    #     bin_list =  format(i, f'0{imgs_num}b')
    #     line_group = []
    #     for j in range(imgs_num):
    #         line_group.append(lines_array[j][int(bin_list[j])])
    #     line_groups.append(line_group)
    
    # shortest_d = np.inf
    # best_line_group = []
    # best_center = [0,0,0]
    # for line_group in line_groups:
        d_sum = 0
        # center_point = ecd.find_closest_point_to_lines(line_group)
        center_point = ecd.find_closest_point_to_lines(lines_array)
        # for line in line_group:
        for line in lines_array:
            d,_ = ecd.calc_point_line_distance(center_point, line)
            d_sum += d
        # if d_sum < shortest_d:
        #     shortest_d = d_sum
        best_line_group = lines_array
        best_center = center_point
    
    # d_sum = 0
    # center_point = ecd.find_closest_point_to_lines(lines_array)
    # for line in lines_array:
    #     d,_ = ecd.calc_point_line_distance(center_point, line)
    #     d_sum += d
        
    # print(center_point)
    # print(d_sum)
    
    return best_center, result_imgs, best_line_group, d_sum

def find_ellipse_center_3d_new(img, depth_img, rots, trans):
    K = normalizer.get_intrinsic_matrix(img)
    result_img, center, _, contour_img, debug = find_ellipse_center_2d(img)
    L = ecd.get_world_line_params(center, K, rots, trans)
    center = center.astype(int)
    depth = depth_img[center[1], center[0]]
    print(depth)
    center_3d = L[0] + depth*L[1]
    return center_3d, result_img, contour_img, L, debug

# def debug_plot_3d_lines(lines, point):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
    
#     colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
#     for i in range(len(lines)):
#         line = generate_line_points(lines[i])
#         ax.plot(line[0], line[1], line[2], label='3D Line', color=colors[i])

#     xp,yp,zp = point
#     ax.scatter(xp, yp, zp, c='r', marker='o')
        
    
    
#     # Set labels for the axes
#     ax.set_title('Reel center estimation in 3D')
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.grid(True)
#     ax.set_box_aspect([1, 1, 1])

#     # Display the plot
#     plt.show()
'''
good result: pos(0,1) pos(2,3)
'''

if __name__ == "__main__":
    path = 'test_sample'
    imgs = []
    rots_array = []
    trans_array = []
    result_imgs = []
    import os
    if os.path.isdir(path):
        files = os.listdir(path)
        for file in files:
            file_path = path + '/' + file
            if os.path.isfile(file_path) and file_path.endswith('.txt'):
                rots, trans = tr.read_transform_data(file_path)
                rots_array.append(rots)
                trans_array.append(trans)
            elif os.path.isfile(file_path) and file_path.endswith('.png'):
                img = cv2.imread(file_path)
                imgs.append(img)
    # import time
    # start = time.time()
    # center_point, lines_array, result_imgs = find_ellipse_center_3d(imgs, rots_array, trans_array)
    # print(f'time:{time.time()-start}')
    
    # import matplotlib.pyplot as plt
    # # fig1 = plt.figure()
    # # ax1 = fig1.add_subplot(121)
    # # ax1.imshow(cv2.cvtColor(result_imgs[0],cv2.COLOR_BGR2RGB))
    # # ax2 = fig1.add_subplot(122)
    # # ax2.imshow(cv2.cvtColor(result_imgs[1],cv2.COLOR_BGR2RGB))

    
    # fig2 = plt.figure()
    # ax = fig2.add_subplot(111, projection='3d')
    
    # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    # # for lines in lines_array:
    # for i in range(len(lines_array)):
    #     lines = lines_array[i]
    #     for j in range(len(lines)): # for lines_array = [group1[L1,L2...],group2[]]
    #         # line = generate_line_points(lines)
    #         line = generate_line_points(lines[j])
    #         ax.plot(line[0], line[1], line[2], label='3D Line', color=colors[j])

    # xp,yp,zp = center_point
    # ax.scatter(xp, yp, zp, c='r', marker='o')
        
    
    
    # # Set labels for the axes
    # ax.set_title('Reel center estimation in 3D')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.grid(True)
    # ax.set_box_aspect([1, 1, 1])

    # # Display the plot
    # plt.show()
    pass
