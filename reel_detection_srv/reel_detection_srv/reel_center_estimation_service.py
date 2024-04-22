
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
        imgs = self.convert_imsgs_to_imgs(request.img_msgs)
        rots_array = [np.array(request.rot1), np.array(request.rot2)]
        trans_array = [np.array(request.pos1),np.array(request.pos2)]
        print('Processing...')
        best_center, result_imgs, best_line_group, d_sum = rdm.orient_find_ellipse_center_3d(imgs, rots_array, trans_array)
        print('Done.')
        result_img_msgs = []
        for img in result_imgs:
            img_msg = bridge.cv2_to_imgmsg(img)
            result_img_msgs.append(img_msg)
        
        response.center_point = best_center.tolist()
        response.result_img_msgs = result_img_msgs
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