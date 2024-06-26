import numpy as np
from numpy import linalg as la
import random
from scipy.linalg import eig

def orient_ellipse_fit(orient_points): # N x 4
    # numerically stable version
    # Ax^2+By^2+Cxy -> Ax^2+Bxy+Cy^2...
    point_num = len(orient_points)
    K = np.array([[0,   0, -2],
                  [0,   1,  0],
                  [-2,  0,  0]])
    H1 = np.empty((0, 3)) # 3n x 3
    H2 = np.empty((0, 3)) # 3n x 3
    H3 = np.empty((0, point_num)) # 3n x n
    
    for i in range(point_num):
        x,y,nx,ny = orient_points[i,:]
        h1 = np.array([[x**2,    x*y,    y**2],
                       [2*x,     y,         0],
                       [0,       x,      2*y,]]) 
        h2 = np.array([[x,  y,  1],
                       [1,  0,  0],
                       [0,  1,  0]])
        h3 = np.zeros((3, point_num))
        h3[1, i] = -nx
        h3[2, i] = -ny
        
        H1 = np.vstack((H1, h1))
        H2 = np.vstack((H2, h2))
        H3 = np.vstack((H3, h3))
    
    L = np.eye(3*point_num) - H3 @ H3.T
    A = H1.T @ (L @ H2 @ np.linalg.inv(H2.T @ L @ H2) @ H2.T - np.eye(3*point_num)) @ L @ H1
    evals, evecs = eig(A, K)
    # evals, evecs = np.linalg.eig(np.linalg.inv(K) @ A)
    v1 = evecs[:, np.argmin(abs(evals))]
    v2 = -np.linalg.inv(H2.T @ L @ H2) @ H2.T @ L @ H1 @ v1

    coeffs = np.append(v1, v2)
    if isinstance(coeffs[0], np.complex128):
        return []
    return coeffs

def check_orient_diffs(coeffs, orient_points, max_angle = np.pi/36):
    if len(coeffs) == 0:
        return False
    
    A, B, C, D, E, F = coeffs
    points = orient_points[:, :2]
    orients = orient_points[:, 2:]
    hom_points = np.hstack((points, np.ones((len(points),1))))
    grad_mat = np.array([[2*A, B],
                         [B, 2*C],
                         [D,   E]])
    new_orients = hom_points @ grad_mat
    norms = np.linalg.norm(new_orients, axis=1)
    new_orients = new_orients / norms[:, np.newaxis]
    angle_diffs = np.arccos(np.sum(orients * new_orients, axis=1))
    angle_diffs = np.where(angle_diffs > np.pi/2, np.pi - angle_diffs, angle_diffs)
    avg_diffs = np.average(angle_diffs)
    # print(angle_diffs)
    if avg_diffs > max_angle:
        return False
    return True

# def points_on_model(coeffs, orient_points, threshold = 0.004):
def points_on_model(coeffs, points, inlier_threshold):            
    points = points[:, :2]
    dists = []
    for i in range(len(points)):
        point = points[i,:]
        _, distance, _ = calc_distance(coeffs, point)
        if distance > inlier_threshold:
            return False
        dists.append(distance)
    # print('point dists:', dists)
    return True
    
    
def get_sample_indices(points_num, sample_size, min_dist = 3):
    sample_indices = []
    while True:
        sample_indices = random.sample(range(points_num), sample_size)
        sample_sort = sorted(sample_indices)
        diffs = np.diff(sample_sort)
        min_diff = np.min(diffs)
        if min_diff >= min_dist:
            break
    return sample_indices

def orient_get_sample_indices(orient_points, points_num, sample_size, min_angle_diff = np.pi/18):
    sample_indices = []
    orients = orient_points[:, 2:]
    counter = 0
    while True:
        if counter > 100:
            return []
        sample_indices = random.sample(range(points_num), sample_size)
        sample_orients = orients[sample_indices]
        # sample_orients2 = np.vstack((sample_orients[1:], sample_orients[0]))
        angles = np.array([np.arccos(np.dot(sample_orients[0],sample_orients[1])),
                           np.arccos(np.dot(sample_orients[1],sample_orients[2])),
                           np.arccos(np.dot(sample_orients[2],sample_orients[0]))])
        # diffs = np.linalg.norm(sample_orients2 - sample_orients, axis=1)
        # if len(diffs[diffs < min_angle_diff]) == 0:
        #     break
        if len(angles[angles < min_angle_diff]) == 0:
            break
        counter += 1
    return sample_indices
    
def cart_to_pol(coeffs):
    """
        Convert the cartesian conic coefficients, (a, b, c, d, e, f), to the
        ellipse parameters, where F(x, y) = ax^2 + bxy + cy^2 + dx + ey + f = 0.
    Return:
        x0, y0, ap, bp, e, phi.
    (x0, y0): ellipse center;
    (ap, bp): the semi-major and semi-minor axes respectively; 
    e: the eccentricity; 
    phi: the rotation of the semi-major axis from the x-axis.
    """

    # We use the formulas from https://mathworld.wolfram.com/Ellipse.html
    # which assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
    # Therefore, rename and scale b, d and f appropriately.
    if len(coeffs) != 6:
        return 0,0,0,0,0,0
    
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]

    den = b**2 - a*c
    if den >= 0:
        return 0,0,0,0,0,0
        # raise ValueError('coeffs do not represent an ellipse: b^2 - 4ac must be negative!')

    # The location of the ellipse centre.
    x0, y0 = (c*d - b*f) / den, (a*f - b*d) / den

    num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
    fac = np.sqrt((a - c)**2 + 4*b**2)
    if fac - a - c == 0 or -fac - a - c == 0:
        return 0,0,0,0,0,0
    # The semi-major and semi-minor axis lengths (these are not sorted).
    ap = num / den / (fac - a - c)
    bp = num / den / (-fac - a - c)
    if ap <= 0 or bp <= 0:
        return 0,0,0,0,0,0
    ap = np.sqrt(ap)
    bp = np.sqrt(bp)

    # Sort the semi-major and semi-minor axis lengths but keep track of
    # the original relative magnitudes of width and height.
    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap

    # The eccentricity.
    r = (bp/ap)**2
    if r > 1:
        r = 1/r
    e = np.sqrt(1 - r)

    # The angle of anticlockwise rotation of the major-axis from x-axis.
    if b == 0:
        phi = 0 if a < c else np.pi/2
    else:
        phi = np.arctan((2.*b) / (a - c)) / 2
        # phi = np.arctan2((2.*b), (a - c)) / 2
        if a > c:
            phi += np.pi/2
    if not width_gt_height:
        # Ensure that phi is the angle to rotate to the semi-major axis.
        phi += np.pi/2
    phi = phi % np.pi

    return x0, y0, ap, bp, e, phi

def ellipse_fit(points):

    N = len(points)
    ret = np.zeros(6)

    #normalize coordinates:
    centerX = np.mean(points[:,0])
    centerY = np.mean(points[:,1])

    total = np.sum(np.square(points[:,0] - centerX) + np.square(points[:,1] - centerY))
    total = np.sqrt(total / N)
    scale = np.sqrt(2.0) / total

    Ts = np.eye(3, dtype=np.float32)
    Ts[0,0] = Ts[1,1] = scale

    Td = np.eye(3, dtype=np.float32)
    Td[0,2] = -centerX
    Td[1,2] = -centerY

    T = Ts @ Td
    fu = T[0,0]
    fv = T[1,1]
    u0 = T[0,2]
    v0 = T[1,2]

    # Construction of scattered matrix:
    D1 = np.zeros((N,3), dtype=np.float32)
    D2 = np.zeros((N,3), dtype=np.float32)
    for idx in range(N):
        x = fu * points[idx,0] + u0
        y = fv * points[idx,1] + v0
        D1[idx,:] = [x*x, x*y, y*y]
        D2[idx,:] = [x, y, 1.0]

    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2

    S = np.zeros((6,6), dtype=np.float32)
    S[:3,:3] = S1
    S[:3,3:] = S2
    S[3:,:3] = S2.T
    S[3:,3:] = S3

    C1 = np.zeros((3,3), dtype=np.float32)
    C1[0,2] = 2.0
    C1[2,0] = 2.0
    C1[1,1] = -1.0

    if la.det(S3) == 0:
        return ret
    
    M = la.inv(C1) @ (S1 - S2 @ la.inv(S3) @ S2.T)
    
    if np.isnan(M).any():
        return ret

    vals, vecs = la.eig(M)

    # parameter selection
    bestError = np.inf
    bestParams = np.empty((0,0))

    for idx in range(3):
        if np.imag(vals[idx]) == 0.0:
            val = np.real(vals[idx])
            a = np.real(vecs[0,idx])
            b = np.real(vecs[1,idx])
            c = np.real(vecs[2,idx])
            
            length = np.sqrt(a*a + b*b + c*c)
            a /= length
            b /= length
            c /= length
            
            check = 4*a*c - b*b
            if check > 0.0:
                a /= np.sqrt(check)
                b /= np.sqrt(check)
                c /= np.sqrt(check)
                
                params13 = np.zeros((3,1), dtype=np.float32)
                params13[0,0] = a
                params13[1,0] = b
                params13[2,0] = c

                params46 = -1 * la.inv(S3) @ S2.T @ params13

                params = np.zeros((6,1), dtype=np.float32)
                params[:3,0] = params13[:3,0]
                params[3:,0] = params46[:3,0]
                
                # Compute error
                errorMtx = params.T @ S @ params
                error = errorMtx[0,0]
                if error < bestError:
                    bestError = error
                    bestParams = params
    
    

    if bestParams.size != 0:
    
        At = bestParams[0,0]
        Bt = bestParams[1,0]
        Ct = bestParams[2,0]
        Dt = bestParams[3,0]
        Et = bestParams[4,0]
        Ft = bestParams[5,0]

        ret[0] = At*fu*fu
        ret[1] = Bt*fu*fv
        ret[2] = Ct*fv*fv
        ret[3] = 2*At*fu*u0+Bt*fu*v0+Dt*fu
        ret[4] = Bt*fv*u0+2*Ct*fv*v0+Et*fv
        ret[5] = At*u0*u0+Bt*u0*v0+Ct*v0*v0+Dt*u0+Et*v0+Ft

    return ret

def get_ransac_iteration_num(points_num, inliers_num, sample_size, confidence):
    """
    To calculate the max iterations of ransac
    
    Args:
        point_num: Total number of points
        inlier_num: Current highest number of inliers of the iteration, changed when find a better model.
        sample_size: The number of points from which the model can be instantiated. 
        confidence: The requested probability of success
    Return:
        k: max iteration num
    """

    inlier_ratio = inliers_num / points_num
    numerator = np.log(1 - confidence)
    denominator = 1 - np.power(inlier_ratio, sample_size)
    if denominator == 0:
        return 0
    
    denominator = np.log(denominator)

    k = numerator / denominator
    
    if(k < 0 or np.isnan(k)):
        return np.inf
    return int(k+1)

def calc_distance(coeffs, point):
    """
    Calculate closest distance from a point to the ellipse.

    Args:
        coeffs: [a,b,c,d,e,f]
        point: (x,y)
    Return:
        closest_point: [x,y] closest point on the ellipse
        distance: float
        candidates: [[x1,y1],[x2,y2]...] candidates of closest point on the ellipse
    """
    
    A, C, B, D, E, F = coeffs # Au2+Bv2+Cuv+Du+Ev+F=0 not Ax2+Bxy+Cy2+Dx+Ey+F=0

    M = np.array([[A, C/2, D/2], [C/2, B, E/2], [D/2, E/2, F]])
    
    tildeM = M[:2, :2]
    a = M[:2,2]

    u0 = point[0]
    v0 = point[1]

    # Polinomials

    poly_1 = [E*C/4.0-B*D
    /2.0, B*u0-D/2.0-v0*C/2.0, u0]

    poly_2 = [D*C/4.0-E*A/2.0, v0*A-E/2.0-u0*C/2.0, v0]

    poly_3 = [A*B-C*C/4, A+B, 1.0]

    final_poly = M[0,0] * np.convolve(poly_1, poly_1) + 2 * M[0,1] * np.convolve(poly_1, poly_2) + M[1,1] * np.convolve(poly_2, poly_2)

    final_poly = final_poly + 2 * a[0] * np.convolve(poly_1, poly_3) + 2 * a[1] * np.convolve(poly_2, poly_3)

    final_poly = final_poly + F * np.convolve(poly_3, poly_3)

    rs = np.roots(final_poly)
    candidates = point

    closest_point = np.zeros(2)
    distance = la.norm(closest_point - point)

    for idx in range(4):
        lambda_val = rs[idx]

        if np.isreal(lambda_val):
            p1 = poly_1[0] * lambda_val**2 + poly_1[1] * lambda_val + poly_1[2]
            p2 = poly_2[0] * lambda_val**2 + poly_2[1] * lambda_val + poly_2[2]
            p3 = poly_3[0] * lambda_val**2 + poly_3[1] * lambda_val + poly_3[2]

            x = np.array([p1,p2])/p3

            candidates = np.vstack((candidates, x))

            if la.norm(x - point) < distance:
                distance = la.norm(x - point)
                closest_point = x

    return closest_point, distance, candidates

# def are_inliers(points, coeffs, inlier_threshold):
#     for i in range(len(points)):
#         point = np.array([points[i,0], points[i,1]])
#         if calc_distance(coeffs, point)[1] > inlier_threshold:
#             return False
#     return True

def get_inliers(points, coeffs, inlier_threshold):
    """
    Calculate perpendicular distance from points to ellipse.
    Return inlier indices.
    """
    # inliers = np.empty((0,0))
    inliers = []
    if len(coeffs) == 0:
        return inliers
    x0, y0, ap, bp, e, phi = cart_to_pol(coeffs)
    if ap == 0:
        return inliers

    for i in range(len(points)):
        point = np.array([points[i,0], points[i,1]])
        _, distance, _ = calc_distance(coeffs, point)
        if distance <= inlier_threshold:
            inliers.append(i)

    return inliers

def img_ransac_ellipse_fit(points, inlier_threshold, confidence, max_failed_num = 10000, max_iteration = np.inf, least_inliers = 0, max_eccentricity = 1, min_sample_dist = 0):
    """
    Standalone function for direct call
    Args:
        points: Denormalized points in the form of [[x1 y1],[x2 y2],...]
        inlier_threshold: Points be inliers if distance < threshold 
        confidence: [0,1], for max ransac iteration calculation
        max_failed_num: The number of consecutive failures to generate a better model
        max_iteration: The real max iteration is the smaller number of max_iteration and calculated max ransac iteration
        least_inliers: Least numbers of inliers. The model won't be considered if the inliers numbers < least_inliers
    Return:
        best_sample_indices: best sample points for model fitting
        best_inlier_indices: inlier points of the best model
    """
    
    sample_size = 5
    points_num = len(points)
    iteration = 0
    ransac_iteration_num = np.inf
    failed_num = 0
    best_samples = []
    best_inliers = []
    best_coeffs = []

    # Normalize points
    # normalizer = MinMaxScaler(feature_range=(-1,1))
    # normed_points = normalizer.fit_transform(points)

    # K = normalizer.get_normalize_matrix(img)
    # normed_points = normalizer.normalize_img_points(points, K)
    normed_points = points
    
    
    # if max_major_axis == 0:
    #     max_major_axis = min(max(normed_points[:,0]), max(normed_points[:,1]))
    # max_major_axis = 2
    
    if points_num < sample_size:
        return best_coeffs, best_samples, best_inliers

    # for loop get best model (ransac iteration)
    while iteration < np.min([max_iteration, ransac_iteration_num]):

        # choose random points for model fitting
        # sample_indices = random.sample(range(points_num), sample_size)
        sample_indices = get_sample_indices(points_num, sample_size, min_sample_dist)
        coeffs = ellipse_fit(normed_points[sample_indices])
        inlier_indices = get_inliers(normed_points, coeffs, inlier_threshold)

        if len(inlier_indices) < least_inliers or len(inlier_indices) <= len(best_inliers): # or not are_inliers(normed_points[sample_indices], coeffs, inlier_threshold):
            failed_num += 1
        else:
            x0, y0, ap, bp, e, phi = cart_to_pol(coeffs)
            if e > max_eccentricity:# or ap >= max_major_axis:
                failed_num += 1
            else:
                best_samples = points[sample_indices]
                best_inliers = points[inlier_indices]
                best_coeffs = coeffs

                failed_num = 0
                ransac_iteration_num = get_ransac_iteration_num(points_num, len(best_inliers), sample_size, confidence)
        
        
        if failed_num >= max_failed_num:
            # print('FAILED!')
            break
        
        iteration += 1
        # if iteration % 10 == 0: print(iteration)
        # if failed_num % 10 == 0: print('failed num:{}'.format(failed_num))
    # denormalize points and fit ellipse
    if len(best_samples) != 0:
        pass
        # best_coeffs = ellipse_fit(best_samples)
        # best_coeffs = normalizer.denormalize_ellipse_coeffs(best_coeffs, K)

    return best_coeffs, best_samples, best_inliers

def ransac_orient_ellipse_fit(orient_points, inlier_threshold, confidence, max_failed_num = 10000, max_iteration = np.inf, least_inliers = 0, max_eccentricity = 1, min_sample_dist = 0, max_major_axis = np.inf):
    """
    Standalone function for direct call
    Args:
        points: Denormalized points in the form of [[x1 y1],[x2 y2],...]
        inlier_threshold: Points be inliers if distance < threshold 
        confidence: [0,1], for max ransac iteration calculation
        max_failed_num: The number of consecutive failures to generate a better model
        max_iteration: The real max iteration is the smaller number of max_iteration and calculated max ransac iteration
        least_inliers: Least numbers of inliers. The model won't be considered if the inliers numbers < least_inliers
    Return:
        best_sample_indices: best sample points for model fitting
        best_inlier_indices: inlier points of the best model
    """
    
    sample_size = 3
    points_num = len(orient_points)
    iteration = 0
    ransac_iteration_num = np.inf
    failed_num = 0
    best_samples = []
    best_inliers = []
    best_coeffs = []
    
    if points_num < sample_size:
        return best_coeffs, best_samples, best_inliers

    # for loop get best model (ransac iteration)
    while iteration < np.min([max_iteration, ransac_iteration_num]):

        # sample_indices = get_sample_indices(points_num, sample_size, min_sample_dist)
        sample_indices = orient_get_sample_indices(orient_points, points_num, sample_size, min_angle_diff=np.pi/18*3)
        if len(sample_indices) == 0:
            iteration += 1
            continue
        sample_points = orient_points[sample_indices]
        coeffs = orient_ellipse_fit(sample_points)
        # if not points_on_model(coeffs, sample_points):
        #     iteration += 1
        #     continue
        inlier_indices = get_inliers(orient_points, coeffs, inlier_threshold)

        if len(inlier_indices) < least_inliers or len(inlier_indices) <= len(best_inliers):
            failed_num += 1
        else:
            x0, y0, ap, bp, e, phi = cart_to_pol(coeffs)
            if (not check_orient_diffs(coeffs, orient_points[inlier_indices])) or (not points_on_model(coeffs, sample_points, inlier_threshold)) or e > max_eccentricity or ap > max_major_axis:
            # if e > max_eccentricity or ap > max_major_axis:
                failed_num += 1
            else:
                best_samples = orient_points[sample_indices]
                best_inliers = orient_points[inlier_indices]
                best_coeffs = coeffs
                new_coeffs = orient_ellipse_fit(orient_points[inlier_indices])
                new_inlier_indices = get_inliers(orient_points, new_coeffs, inlier_threshold)
                if len(new_inlier_indices) >= len(inlier_indices) and check_orient_diffs(new_coeffs, orient_points[inlier_indices]) and points_on_model(coeffs, sample_points, inlier_threshold):
                # if check_orient_diffs(coeffs, orient_points[inlier_indices]):
                    best_samples = orient_points[inlier_indices]
                    best_inliers = orient_points[new_inlier_indices]
                    best_coeffs = new_coeffs

                failed_num = 0
                ransac_iteration_num = get_ransac_iteration_num(points_num, len(best_inliers), sample_size, confidence)
                # else:
                #     best_samples = orient_points[sample_indices]
                #     best_inliers = orient_points[inlier_indices]
                #     best_coeffs = best_coeffs2
                #     # failed_num += 1
                    
        
        if failed_num >= max_failed_num:
            # print('FAILED!')
            break
        
        iteration += 1
        # if iteration % 100 == 0: print(iteration, ransac_iteration_num)
        # if failed_num % 100 == 0: print('failed num:{}'.format(failed_num))

    if len(best_samples) != 0:
        while len(best_inliers) > len(best_samples):
            best_samples = best_inliers
            best_coeffs = orient_ellipse_fit(best_samples)
            inlier_indices = get_inliers(orient_points, best_coeffs, inlier_threshold)
            best_inliers = orient_points[inlier_indices]
    # check_orient_diffs(best_coeffs, best_inliers)
    # points = best_samples[:, :2]
    # dists = []
    # for i in range(len(points)):
    #     point = points[i,:]
    #     _, distance, _ = calc_distance(best_coeffs, point)
    #     dists.append(distance)
    # print('pe dist:',dists, np.average(dists))
    return best_coeffs, best_samples, best_inliers

if __name__ == "__main__":

    x0 = 1.96031089e-01
    a1 = 0.19493088971688963
    s = x0/a1
    s1 = 1.02457218e+00/np.sqrt(a1**2+1)
    print(s, s1)
    pass
    