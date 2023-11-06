import numpy as np
import random
import cv2

from reel_detection_srv.main_code import ellipse_fitting as ef
from reel_detection_srv.main_code import ellipse_center_detector as ecd
from reel_detection_srv.main_code import image_processor as improc
from reel_detection_srv.main_code import normalizer
from reel_detection_srv.main_code import utils

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

def find_ellipse_center_point(img):
    #ransac params
    inlier_threshold = 0.002
    confidence = 0.95
    max_failed_num = 50 # max consecutive fail times
    max_eccentricity = 0.8
    max_iteration = 500
    least_inliers = 0 # minimum inlier amount to be considered a candidate model
    min_inlier_rate = 0.6
    
    img = improc.preproc(img, output_gray=True)
    
    [h,w] = img.shape[:2]
    result_img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    contours = improc.get_contours(img)
    # contours = improc.simplify_contours(contours, 0.1)

    final_points = np.empty((0,2))
    contour_img = result_img.copy()
    sample_step_size = int(np.ceil(np.log(h*w)))+2

    K = normalizer.get_normalize_matrix(img)
    
    for i in range(len(contours)):
        points = contours[i].squeeze(axis=1)
        points = points[::sample_step_size,:]
        # final_points = np.vstack((final_points, points))
        cv2.drawContours(contour_img, [contours[i]], -1, utils.get_color(i,True), 2)
        
        normed_points = normalizer.normalize_img_points(points, K)
        coeffs, samples, inliers = ef.img_ransac_ellipse_fit(normed_points, inlier_threshold, confidence, max_iteration=max_iteration, max_failed_num=max_failed_num, max_eccentricity=max_eccentricity)

        if len(inliers) / len(points) >= min_inlier_rate:
            final_points = np.vstack((final_points, points))

    #         ellipse_points = get_ellipse_pts(ef.cart_to_pol(coeffs), npts=500).T
    #         contour_img = improc.draw_img_points(contour_img, points, 3, utils.get_color(i,True))
    #         contour_img = improc.draw_img_points(contour_img, ellipse_points, 2, (255,0,0))
    #         contour_img = improc.draw_img_points(contour_img, inliers, 1, (0,255,0))
    # cv2.imshow('contour',contour_img)


    #final fitting
    if len(final_points) == 0:
        # raise SystemExit
        return 0
    # final_points = np.array(final_points)
    final_points = normalizer.normalize_img_points(final_points, K)
    coeffs, samples, inliers = ef.img_ransac_ellipse_fit(final_points, inlier_threshold, confidence, max_failed_num=max_failed_num, max_eccentricity=max_eccentricity, least_inliers=least_inliers)
    
    if len(coeffs) == 0:
        return 0

    beta = random.random() * 2 * np.pi
    H1,H2 = ecd.create_transform_matrix2(coeffs, beta)

    center1 = H1 @ np.array([0,0,1]).T
    center2 = H2 @ np.array([0,0,1]).T

    center1 = center1/center1[-1]
    center2 = center2/center2[-1]

    coeffs = normalizer.denormalize_ellipse_coeffs(coeffs, K)
    samples = normalizer.denormalize_img_points(samples, K)
    inliers = normalizer.denormalize_img_points(inliers, K)

    x0, y0, ap, bp, e, phi = ef.cart_to_pol(coeffs)
    ellipse_points = get_ellipse_pts([x0, y0, ap, bp, e, phi], npts=500).T
    result_img = improc.draw_img_points(result_img, final_points, 3, (255, 0, 0))
    result_img = improc.draw_img_points(result_img, ellipse_points, 2, (0, 128, 255))
    result_img = improc.draw_img_points(result_img, samples, 4, (0,0,255))
    result_img = improc.draw_img_points(result_img, inliers, 1, (0,255,0))
    
    
    center1 = K @ center1
    center2 = K @ center2
    
    result_img = cv2.circle(result_img, (int(center1[0]),int(center1[1])),5,(0,240,240),-1)
    result_img = cv2.circle(result_img, (int(center2[0]),int(center2[1])),5,(0,255,0),-1)

    return result_img

if __name__ == "__main__":
    # img = cv2.imread('ros_data/ros_reel_0_0.png', cv2.IMREAD_GRAYSCALE)
    # result_img = find_ellipse_center_point(img)
    # cv2.imshow('result', result_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    pass