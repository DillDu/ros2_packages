import cv2
import numpy as np
import os
from reel_detection_srv.main_code import utils

def edge_detect_sobel(img):
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x = np.uint8(np.absolute(sobel_x))
    prewitt_y = cv2.filter2D(img, -1, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))
    prewitt_y = np.uint8(np.absolute(prewitt_y))
    edge_img = cv2.bitwise_or(sobel_x, prewitt_y)
    return edge_img

def get_edge_img(img):
    #Blur
    # img = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Canny
    # edge_img = cv2.Canny(sharpen_img(img), 50, 200)
    out_img = cv2.Canny(img, 10, 200)
    
    # Laplacian
    # laplacian = cv2.Laplacian(img, cv2.CV_16S)
    
    #Sobel
    # edge_img = edge_detect_sobel(img)
    
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
        [h,w] = img.shape[:2]
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        edge_img = cv2.Canny(img, 20, 200)
        contours, _ = cv2.findContours(edge_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rm_lines_img = remove_lines(edge_img)

        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

        # edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        edge_direction = np.arctan2(sobel_y, sobel_x)

        out_img = np.zeros_like(img, shape=(h, 2*w))
        out_img[:,:w] = rm_lines_img
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
                if rm_lines_img[y,x]:
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
            cv2.drawContours(out_img, [c], -1, utils.get_color(i,True), 1)
            # step = .5
            
            for j in range(len(c)):
                cv2.circle(out_img, c[j][0], 2, utils.get_color(i,True), -1)
            i = (i+1)
        cv2.imwrite(out_dir + '/' + basename + '_contour' + suffix, out_img)
def generate_contour_img_files(folder_path, out_dir = 'contour_result', min_point_num = 100):
    if os.path.isdir(folder_path):
        files = os.listdir(folder_path)
        for file in files:
            file_path = folder_path + '/' + file
            generate_contour_img_file(file_path, out_dir, min_point_num)

def remove_corners(binary_img, threshold=100):
    block_size = 3
    ksize = 3
    k = 0.05

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
def remove_lines(binary_img, threshold=50):
    
    linesP = cv2.HoughLinesP(binary_img, 1, np.pi / 360, threshold, None, 50, 3)
    # binary_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    if linesP is not None:
        for l in linesP[:,0]:
            cv2.line(binary_img, (l[0], l[1]), (l[2], l[3]), (0,0,0), 2, cv2.LINE_AA) 
    return binary_img        

def simplify_contours(contours, epsilon):
    simplified_contours = []
    for contour in contours:
        # eps = epsilon * cv2.arcLength(contour, True)
        simplified_contour = cv2.approxPolyDP(contour, epsilon, closed=False)
        simplified_contours.append(simplified_contour)
    
    return simplified_contours
    
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
    
def get_oedge_points():
    pass

def get_edge_points(img, lower_threshold=20, upper_threshold=200):
    edge_img = cv2.Canny(img, lower_threshold, upper_threshold)
    points = np.empty((0, 2))
    for y in range(edge_img.shape[0]):
        for x in range(edge_img.shape[1]):
            if edge_img[y,x]:
                points = np.vstack(points, np.array([x,y]))
    return points

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
    img = sharpen_img(img)
    edge_img = get_edge_img(img)
    edge_img = remove_corners(edge_img)
    edge_img = remove_lines(edge_img.astype(np.uint8))

    contours, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    valid_contours = []
    for c in contours:
        if filter and len(c) < min_point_num:
            continue
        epsilon = 0.01 * cv2.arcLength(c, True)
        a = cv2.approxPolyDP(c, epsilon, True)
        A = cv2.contourArea(c)
        P = cv2.arcLength(a, False)
        
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

def draw_img_points(img, points, radius, color):
    for i in range(len(points)):
        x = int(points[i,0])
        y = int(points[i,1])
        cv2.circle(img,(x,y),radius,color,-1)
    return img

if __name__ == '__main__':
    # generate_edge_img_files('reel_data')
    # generate_contour_img_files('reel_data')
    # generate_orientation_edge_imgs('reel_data')
    
    # img = cv2.imread('reel_data/reel_7_4.png',cv2.IMREAD_GRAYSCALE)

    # valid_contours = get_contours(img, filter=True)
    # for c in valid_contours:
    #     for i in range(len(c)):
    #         cv2.circle(img, c[i][0], 2, 255,-1)
            
    # cv2.imshow("win,",img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    pass