
import rclpy
from rclpy.node import Node

import cv2

from reel_detection_interfaces.srv import DetectReelCenter2D
from cv_bridge import CvBridge

from reel_detection_srv.main_code import reel_detection_main as rdm


class ReelCenter2DService(Node):

    def __init__(self):
        super().__init__('reel_fitting_service')
        self.srv = self.create_service(DetectReelCenter2D, 'detect_reel_center_2d', self.detect_reel_center_callback)
    
    def detect_reel_center_callback(self, request, response):
        bridge = CvBridge()
        img = bridge.imgmsg_to_cv2(request.img_msg, 'bgr8')
        result_img = rdm.find_ellipse_center_2d(img)
        response.result_img_msg = bridge.cv2_to_imgmsg(result_img)
        self.get_logger().info('Incoming request received\n' % ())

        return response

def main(args=None):
    rclpy.init(args=args)

    reel_fitting_service = ReelCenter2DService()

    rclpy.spin(reel_fitting_service)

    rclpy.shutdown()

if __name__ == '__main__':
    main()