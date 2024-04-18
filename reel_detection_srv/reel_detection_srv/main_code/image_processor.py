import cv2
import numpy as np
import os
from reel_detection_srv.main_code.my_utils import *
    

def edge_detect_sobel(img):
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x = np.uint8(np.absolute(sobel_x))
    prewitt_y = cv2.filter2D(img, -1, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))
    prewitt_y = np.uint8(np.absolute(prewitt_y))
    edge_img = cv2.bitwise_or(sobel_x, prewitt_y)
    return edge_img

def get_edge_img(img):
    #Blur
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Canny
    # edge_img = cv2.Canny(sharpen_img(img), 50, 200)
    out_img = cv2.Canny(img, 50, 150)
    
    # Laplacian
    # laplacian = cv2.Laplacian(img, cv2.CV_16S)
    
    #Sobel
    # out_img = edge_detect_sobel(img)
    
    # _, threshold = cv2.threshold(laplacian,30,255,cv2.THRESH_BINARY)
    
    # Adaptive threshold
    # out_img = cv2.adaptiveThreshold(out_img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,0)
    # out_img = cv2.medianBlur(out_img, 5)
    # out_img = remove_lines(out_img.astype(np.uint8))
    return out_img

def generate_edge_img_file(file_path, out_dir = 'edge_result'):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if os.path.isfile(file_path) and file_path.endswith(('.png', '.jpg', 'jpeg')):
        [basename, suffix] = os.path.splitext(os.path.basename(file_path))
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img = sharpen_img(img)
        out_img = get_edge_img(img)

        # out_img = remove_corners(out_img)
        

        
        
        # out_img = remove_lines(out_img)
        # out_img = zhangSuenThinning(out_img)
        cv2.imwrite(out_dir + '/' + basename + '_edge' + suffix, out_img)
def generate_edge_img_files(folder_path, out_dir = 'edge_result'):
    if os.path.isdir(folder_path):
        files = os.listdir(folder_path)
        for file in files:
            file_path = folder_path + '/' + file
            generate_edge_img_file(file_path, out_dir)

def generate_oedge_img_file(file_path, out_dir = 'edge_result'):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if os.path.isfile(file_path) and file_path.endswith(('.png', '.jpg', 'jpeg')):
        [basename, suffix] = os.path.splitext(os.path.basename(file_path))
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        [h,w] = img.shape[:2]
        img = sharpen_img(img)
        # blurred = img
        edge_img = get_edge_img(img)
        edge_img = remove_corners(edge_img)
        edge_img = remove_lines(edge_img)
        # edge_img = cv2.Canny(img, 20, 200)
        # contours, _ = cv2.findContours(edge_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # rm_lines_img = remove_lines(edge_img)

        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

        # edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        edge_direction = np.arctan2(sobel_y, sobel_x)

        out_img = np.zeros_like(img, shape=(h, 2*w))
        out_img[:,:w] = edge_img
        out_img = cv2.cvtColor(out_img, cv2.COLOR_GRAY2BGR)
        step = 20

        i = 0
        for c in contours:
            # cv2.drawContours(img, [c], -1, colors_list[i], 2)
            # i = (i+1)%len(colors_list)
            if len(c) < 20:
                continue
            for i in range(0,len(c),step):
                [x,y] = c[i][0]
                # print(c[i])
                if edge_img[y,x]:
                    angle_rad = edge_direction[y, x]
                    length = 5  # Length of the orientation line (adjust as needed)
                    x1 = int(w+x + length * np.cos(angle_rad+np.pi/2))
                    y1 = int(y + length * np.sin(angle_rad+np.pi/2))
                    x2 = int(w+x + length * np.cos(angle_rad-np.pi/2))
                    y2 = int(y + length * np.sin(angle_rad-np.pi/2))
                    cv2.line(out_img, (x1, y1), (x2, y2), (0,0,255), 1)
                    out_img[y,w+x] = (255,255,255)

        cv2.imwrite(out_dir + '/' + basename + '_oedge' + suffix, out_img)
def generate_oedge_img_files(folder_path, out_dir = 'edge_result'):
    if os.path.isdir(folder_path):
        files = os.listdir(folder_path)
        for file in files:
            file_path = folder_path + '/' + file
            generate_oedge_img_file(file_path, out_dir)

def generate_contour_img_file(file_path, out_dir = 'contour_result', min_point_num = 50):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if os.path.isfile(file_path) and file_path.endswith(('.png', '.jpg', 'jpeg')):
        [basename, suffix] = os.path.splitext(os.path.basename(file_path))
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
        contours = get_contours(img, min_point_num)
        # contours, _ = cv2.findContours(edge_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = simplify_contours(contours, 0.02)
    
        out_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # out_img = img
        i = 0
        for c in contours:
            cv2.drawContours(out_img, [c], -1, get_color(i,True), 1)
            # step = .5
            
            for j in range(len(c)):
                cv2.circle(out_img, c[j][0], 2, get_color(i,True), -1)
            i = (i+1)
        cv2.imwrite(out_dir + '/' + basename + '_contour' + suffix, out_img)
def generate_contour_img_files(folder_path, out_dir = 'contour_result', min_point_num = 100):
    if os.path.isdir(folder_path):
        files = os.listdir(folder_path)
        for file in files:
            file_path = folder_path + '/' + file
            generate_contour_img_file(file_path, out_dir, min_point_num)

def remove_corners(binary_img, threshold=100):
    block_size = 5
    ksize = 3
    k = 0.04

    # Apply Harris corner detection
    corners = cv2.cornerHarris(binary_img, blockSize=block_size, ksize=ksize, k=k)

    # Normalize the corner response
    corners = cv2.normalize(corners, None, 0, 255, cv2.NORM_MINMAX)

    # Threshold the corner response to detect corners
    threshold = 110
    corner_coords = np.column_stack(np.where(corners > threshold))
    for x,y in corner_coords:
        cv2.circle(binary_img, (y,x), 3, 0, -1)
    return binary_img
def remove_lines(binary_img, threshold=50, minLineLength=50, maxLineGap=3):
    
    linesP = cv2.HoughLinesP(binary_img, 0.5, np.pi / 180, threshold, None, minLineLength, maxLineGap)
    # binary_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    if linesP is not None:
        for l in linesP[:,0]:
            cv2.line(binary_img, (l[0], l[1]), (l[2], l[3]), (0,0,0), 2, cv2.LINE_4) 
    return binary_img        

def simplify_contours(contours, epsilon):
    simplified_contours = []
    for contour in contours:
        # eps = epsilon * cv2.arcLength(contour, True)
        simplified_contour = cv2.approxPolyDP(contour, epsilon, closed=False)
        simplified_contours.append(simplified_contour)
    
    return simplified_contours

def get_contour_points(contour, sample_step_size):
    points = contour.squeeze(axis=1)
    points = points[::sample_step_size,:]
    points = points[::2,:]
    return points

def get_img_orient(img, ksize = 3):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blurred = sharpen_img(img)
    # [h,w] = img.shape[:2]
    # blurred = cv2.GaussianBlur(img, (3, 3), 0)
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    edge_direction = np.arctan2(sobel_y, sobel_x)

    return edge_direction
    
def sharpen_img(img, blur = True, blur_size = 3):
    # img = cv2.equalizeHist(img)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # img = clahe.apply(img)

    # sharpen
    kernel = np.array([[0, -1, 0], 
                       [-1, 5, -1], 
                       [0, -1, 0]], dtype=np.float32)
    ret_img = cv2.filter2D(img, -1, kernel)
    # blur
    if blur:
        ret_img = cv2.GaussianBlur(img,(blur_size,blur_size),0)
    return ret_img
def preproc(img, output_gray = False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue_shift = -50
    saturation_factor = 1.5
    img[:, :, 0] = (img[:, :, 0] + hue_shift) % 180
    img[:, :, 1] = np.clip(img[:, :, 1] * saturation_factor, 0, 255)
    
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    if output_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img 

def get_orientation_edge_points(img, sample_interval=20, min_contour_points=20):
    # [h,w] = img.shape[:2]
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    edge_img = cv2.Canny(blurred, 20, 200)
    contours, _ = cv2.findContours(edge_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rm_lines_img = remove_lines(edge_img)

    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    # edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    edge_direction = np.arctan2(sobel_y, sobel_x)

    orient_points = np.empty((0,3))
    i = 0
    for c in contours:
        # cv2.drawContours(img, [c], -1, colors_list[i], 2)
        # i = (i+1)%len(colors_list)
        if len(c) < min_contour_points:
            continue
        for i in range(0,len(c),sample_interval):
            [x,y] = c[i][0]
            # print(c[i])
            if rm_lines_img[y,x]:
                angle_rad = edge_direction[y, x]
                orient_points = np.vstack((orient_points, np.array([x,y,angle_rad])))
    
    return orient_points

def get_contours(img, min_point_num = 50, filter = True):
    # img = sharpen_img(img)
    edge_img = get_edge_img(img)
    edge_img = remove_corners(edge_img)
    # edge_img = remove_lines(edge_img.astype(np.uint8))
    edge_img = remove_lines(edge_img)

    contours, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    for c in contours:
        # if filter and len(c) < min_point_num:
        #     continue
        epsilon = 0.01 * cv2.arcLength(c, True)
        a = cv2.approxPolyDP(c, epsilon, True)
        A = cv2.contourArea(c)
        P = cv2.arcLength(a, False)
        length = cv2.arcLength(c, False)
        box=cv2.minAreaRect(c)
        box=cv2.boxPoints(box).reshape((4,1,2))
        box_length = cv2.arcLength(box, True)

        if filter:
            # if A < 400:
            #     continue
            # if a.shape[0] == 2:
            #     continue
            if len(a) < 6:
                continue
            if P < 100:
                continue
            
        
        # points = c.reshape(-1,2)
        # points = points[::3]
        # valid_contours.append(points)
        
        valid_contours.append(c)

        
    return valid_contours

def get_shattered_contours(img, filter = True):
    # img = sharpen_img(img)
    edge_img = get_edge_img(img)
    edge_img = remove_corners(edge_img)
    # edge_img = remove_lines(edge_img.astype(np.uint8))
    edge_img = remove_lines(edge_img)

    pre_contours, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    for c in pre_contours:
        epsilon = 0.1 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        P = cv2.arcLength(approx, False)
        if approx.shape[0] >= 2:
            for point in approx:
                x, y = point.ravel()
                cv2.circle(edge_img, [x,y], 1, 0, -1)
        
    contours, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        # if filter and len(c) < min_point_num:
        #     continue
        epsilon = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        # A = cv2.contourArea(c)
        P = cv2.arcLength(approx, False)
        # length = cv2.arcLength(c, False)
        # box=cv2.minAreaRect(c)
        # box=cv2.boxPoints(box).reshape((4,1,2))
        # box_length = cv2.arcLength(box, True)
            
        if filter:
            # if A < 400:
            #     continue
            # if a.shape[0] == 2:
            #     continue
            if len(approx) < 6:
                continue
            if P < 100:
                continue
            
        
        # points = c.reshape(-1,2)
        # points = points[::3]
        # valid_contours.append(points)
        
        valid_contours.append(c)
        
    return valid_contours

def draw_points(img, points, radius, color, x_mark = False):
    for i in range(len(points)):
        x = int(points[i,0])
        y = int(points[i,1])
        if x_mark:
            cv2.drawMarker(img, (x,y), color, markerType=cv2.MARKER_CROSS, markerSize=radius*2, thickness=1)
        else:
            cv2.circle(img,(x,y),radius,color,-1)
    # return img

def draw_orient_points(img, orient_points, line_length = 5, line_thickness = 1):
    for i in range(len(orient_points)):
        x,y,nx,ny = orient_points[i,:]
        rad = np.arctan2(ny,nx)
        x1 = int(x + line_length * np.cos(rad))
        y1 = int(y + line_length * np.sin(rad))
        x2 = int(x - line_length * np.cos(rad))
        y2 = int(y - line_length * np.sin(rad))
        x = int(x)
        y = int(y)
        cv2.line(img, (x1, y1), (x2, y2), (0,0,255), line_thickness)
        # cv2.line(img, (x1, y1), (x, y), (0,0,255), 1)
        img[y, x] = (255,255,255)
    # return img

def remove_near_points(points, k = 10):
    i = 0
    while i < len(points):
        p = points[i,:]
        diffs = points - p
        norms = np.linalg.norm(diffs[:,:2], axis=1)
        points = points[norms >= k]
        points = np.vstack((p, points))
        i += 1
        
    return points

def draw_contours(img, contours):
    for i, c in enumerate(contours):
        cv2.drawContours(img, [c], 0, get_color(i), 2)

"""
Image processing method below from paper: Matching 2-D Ellipses to 3-D ircles with Application to Vehicle Pose Identification
"""

def smooth_image(gray_img):
    if len(gray_img.shape) > 2:
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
    [h,w] = gray_img.shape[:2]
    short_side = min(h,w)
    window_size = int(0.1*short_side)
    if window_size % 2 == 0:
        window_size += 1
    half_window = window_size // 2
    smoothed = np.zeros_like(gray_img)
    # smoothing
    for y in range(half_window, h - half_window):
        for x in range(half_window, w - half_window):
            window = gray_img[y - half_window : y + half_window + 1, x - half_window : x + half_window + 1]
            mean_value = cv2.mean(window)
            smoothed[y, x] = mean_value[0]
    # fill corners
    smoothed[:half_window, :half_window] = smoothed[half_window, half_window]
    smoothed[:half_window, w - half_window:w] = smoothed[half_window, w - half_window - 1]
    smoothed[h - half_window:h, :half_window] = smoothed[h - half_window - 1, half_window]
    smoothed[h - half_window:h, w - half_window:w] = smoothed[h - half_window - 1, w - half_window - 1]
    # top
    smoothed[:half_window, half_window:w - half_window] = np.tile(smoothed[half_window, half_window:w - half_window], (half_window, 1))
    # bottom
    smoothed[h - half_window:h, half_window:w - half_window] = np.tile(smoothed[h - half_window - 1, half_window:w - half_window], (half_window, 1))
    # left
    smoothed[half_window:h - half_window, :half_window] = np.tile(smoothed[half_window:h - half_window, half_window], (half_window, 1)).T
    # right
    smoothed[half_window:h - half_window, w - half_window:w] = np.tile(smoothed[half_window:h - half_window, w - half_window - 1], (half_window, 1)).T

    # result_matrix = np.uint8(result_matrix)
    return smoothed

def get_channel_images(img):
    channel_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), img[:,:,0], img[:,:,1], img[:,:,2]]
    return channel_imgs

def get_binary_image(gray_img, smoothed_img, threshold = 150):
    normed = gray_img - smoothed_img
    _, binary_img = cv2.threshold(normed, threshold, 255, cv2.THRESH_BINARY)
    return binary_img

def get_binary_images(channel_imgs, smoothed_img, thresh = 150):
    binary_imgs = []
    for i in range(len(channel_imgs)):
        channel_img = channel_imgs[i]
        # smoothed = smoothed_imgs[i]
        normed = channel_img - smoothed_img
        _, binary_img = cv2.threshold(normed, thresh, 255, cv2.THRESH_BINARY)
        # _, inv_binary_img = cv2.threshold(normed, thresh, 255, cv2.THRESH_BINARY_INV)
        # inv_binary_img = cv2.bitwise_not(binary_img)
        # binary_imgs.extend([binary_img, inv_binary_img])
        binary_imgs.append(binary_img)
    return binary_imgs

def rough_locate_ellipse(binary_imgs):
    best_ratio = np.inf
    best_contour = np.zeros_like(binary_imgs[0])
    for binary_img in binary_imgs:
        contour_image = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        # best_contour = []
        for i, contour in enumerate(contours):
            # color = colors[i % len(colors)]  # Use modulo to cycle through colors
            if len(contour) < 5:
                continue
            ratio, coeff = is_ellipse(binary_img, contour)
            
            if ratio >= 0:
                cv2.drawContours(contour_image, [contour], -1, (0,255,0), 2)
                cv2.ellipse(contour_image, [int(x) for x in coeff[0]], [int(x/2) for x in coeff[1]], coeff[2], 0, 360, (255,0,0), 1)
                if ratio < best_ratio:
                    best_ratio = ratio
                # best_contour = contour
                    best_contour = contour_image
        # if len(best_contour) > 0:
        #     cv2.drawContours(contour_image, [contour], -1, (0,255,0), 2)
        #     best_contour_img = contour_image
    print(best_ratio)
    return best_contour
    # return np.zeros_like(binary_imgs[0])
        # cv2.imshow('im', contour_image)
        # cv2.waitKey()
    
def is_ellipse(img, contour, aspect_ratio_threshold = 0.1, area_ratio_threshold = 0.1):
    
    # Calculate contour area
    contour_area = cv2.contourArea(contour)
    if contour_area == 0 or contour_area < 2000:
        return -1, []
    
    # Fit an ellipse to the contour
    ellipse = cv2.fitEllipse(contour)
    center, axes, angle = ellipse
    if max(axes[0], axes[1]) >= np.max(img.shape[:2]):
        return -1, []
    
    ellipse_area = np.pi * axes[0] * axes[1] / 4
    aspect_ratio = axes[0] / axes[1]

    ellipse_mask = np.zeros_like(img)
    ellipse_mask = cv2.ellipse(ellipse_mask, [int(x) for x in center], [int(y/2) for y in axes], angle, 0, 360, 255, thickness=-1)
    ellipse_img_area = np.sum(ellipse_mask == 255)
    
    contour_mask = np.zeros_like(img)
    cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
    
    
    ref_contour_img = contour_mask.copy()
    # cv2.drawContours(ref_contour_img, [contour], -1, 255, thickness=cv2.FILLED)
    ref_contour_img[ellipse_mask == 255] = 0
    contour_rest_area = np.sum(ref_contour_img == 255)
    contour_area_ratio = contour_rest_area / contour_area
    
    # ref_ellipse_img = np.zeros_like(img)
    # ref_ellipse_img = cv2.ellipse(ref_ellipse_img, [int(x) for x in center], [int(y/2) for y in axes], angle, 0, 360, 255, thickness=-1)
    ellipse_mask[contour_mask == 255] = 0
    ellise_rest_area = np.sum(ellipse_mask == 255)
    ellipse_area_ratio = ellise_rest_area / ellipse_area
    
    avg_area_ratio = (contour_area_ratio + ellipse_area_ratio)/2
    
    ellipse_lost_ratio = (ellipse_area - ellipse_img_area)/ellipse_area
    print(avg_area_ratio, ellipse_lost_ratio)
    # if (aspect_ratio > aspect_ratio_threshold and
    #         contour_area_ratio < area_ratio_threshold and ellipse_area_ratio < area_ratio_threshold):
    if (aspect_ratio > aspect_ratio_threshold and avg_area_ratio < area_ratio_threshold and ellipse_lost_ratio < 0.002):
        # print(rest_area, contour_area, area_ratio)
        
        return avg_area_ratio, ellipse
    else:
        return -1, []

if __name__ == '__main__':
    # generate_edge_img_files('reel_data')
    # generate_contour_img_files('reel_data')
    # generate_oedge_img_files('test_sample')
    
    files = get_file_paths('reels/reel_data', 'png')
    for f in files:
        img = cv2.imread(f)
        # smoothed = smooth_image(img)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # normed = gray - smoothed
        # edge = cv2.Canny(normed, 50, 150)
        edge = get_edge_img(img)
        cv2.imshow(f,edge)
        cv2.waitKey()
    # channel_imgs = get_channel_images(img)
    # binary_imgs = get_binary_images(channel_imgs, smoothed)
    # for b in binary_imgs:
    #     cv2.imshow('bi', b)
    #     cv2.waitKey()
    
    # contour_img = rough_locate_ellipse(binary_imgs)
    # cv2.imshow('i', img)
    # cv2.imshow('img', contour_img)
    # cv2.waitKey()
    
    
    pass