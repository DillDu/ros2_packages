
import sys
import rclpy
from rclpy.node import Node

import cv2

from reel_detection_interfaces.srv import DetectReelCenter
from cv_bridge import CvBridge


class ReelFittingClientAsync(Node):

    def __init__(self):
        super().__init__('reel_fitting_client_async')
        self.cli = self.create_client(DetectReelCenter, 'detect_reel_center_2d')       
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = DetectReelCenter.Request()                                   

    def send_request(self):
        bridge = CvBridge()
        path = sys.argv[1]
        img = cv2.imread(path)
        img_msg = bridge.cv2_to_imgmsg(img, "bgr8")  # "bgr8" is the encoding format, adjust as needed
        self.req.img_msg = img_msg
        self.future = self.cli.call_async(self.req)


def main(args=None):
    rclpy.init(args=args)

    reel_fitting_client = ReelFittingClientAsync()
    reel_fitting_client.send_request()

    while rclpy.ok():
        rclpy.spin_once(reel_fitting_client)
        if reel_fitting_client.future.done():
            try:
                response = reel_fitting_client.future.result()
            except Exception as e:
                reel_fitting_client.get_logger().info(
                    'Service call failed %r' % (e,))
            else:
                reel_fitting_client.get_logger().info(
                    'Result of ellipse_fitting: ')
                bridge = CvBridge()
                img = bridge.imgmsg_to_cv2(response.result_img_msg)
                cv2.imshow('img',img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            break

    reel_fitting_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()