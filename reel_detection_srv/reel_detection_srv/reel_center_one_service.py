
import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge
import numpy as np

from reel_detection_interfaces.srv import DetectReelCenter3D

from reel_detection_srv.main_code import reel_detection_main as rdm


class ReelCenterEstimationService(Node):

    def __init__(self):
        super().__init__('reel_center_estimation_service')
        self.srv = self.create_service(DetectReelCenter3D, 'detect_reel_center_3d', self.detect_reel_center_callback)
    
    def detect_reel_center_callback(self, request, response):
        bridge = CvBridge()
        img = bridge.imgmsg_to_cv2(request.img_msg, 'bgr8')
        rots = np.array(request.rot)
        trans = np.array(request.pos)
        depth = bridge.imgmsg_to_cv2(request.depth_msg, desired_encoding='passthrough')
        
        # best_center, best_line_group, result_imgs = rdm.find_ellipse_center_3d(imgs, rots_array, trans_array)
        best_center, result_img, _,_,_ = rdm.find_ellipse_center_3d_new(img, depth, rots, trans)
        
        # result_img_msgs = []
        # for img in result_imgs:
        #     img_msg = bridge.cv2_to_imgmsg(img)
        #     result_img_msgs.append(img_msg)
        result_img_msg = bridge.cv2_to_imgmsg(result_img)
        
        response.center_point = best_center.tolist()
        response.result_img_msg = result_img_msg
        self.get_logger().info('Incoming request received\n' % ())

        return response
    
    def convert_imsgs_to_imgs(self, img_msgs):
        bridge = CvBridge()
        imgs = []
        for msg in img_msgs:
            img = bridge.imgmsg_to_cv2(msg, 'bgr8')
            imgs.append(img)
        return imgs

def main(args=None):
    rclpy.init(args=args)

    reel_center_estimation_service = ReelCenterEstimationService()

    rclpy.spin(reel_center_estimation_service)

    rclpy.shutdown()

if __name__ == '__main__':
    main()