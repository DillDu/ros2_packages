
import sys
import rclpy
from rclpy.node import Node

import cv2
import os

from reel_detection_interfaces.srv import DetectReelCenter3D
from cv_bridge import CvBridge


class ReelCenter3DClientAsync(Node):

    def __init__(self):
        super().__init__('center_estimation_3d_client')
        self.cli = self.create_client(DetectReelCenter3D, 'detect_reel_center_3d')       
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = DetectReelCenter3D.Request()                                   

    def send_request(self):
        bridge = CvBridge()
        path = sys.argv[1]
        img = cv2.imread(path)
        img_msg = bridge.cv2_to_imgmsg(img, "bgr8")  # "bgr8" is the encoding format, adjust as needed
        self.req.img_msg = img_msg
        self.future = self.cli.call_async(self.req)


def main(args=None):
    rclpy.init(args=args)

    reel_center_3d_client = ReelCenter3DClientAsync()
    reel_center_3d_client.send_request()

    while rclpy.ok():
        rclpy.spin_once(reel_center_3d_client)
        if reel_center_3d_client.future.done():
            try:
                response = reel_center_3d_client.future.result()
            except Exception as e:
                reel_center_3d_client.get_logger().info(
                    'Service call failed %r' % (e,))
            else:
                reel_center_3d_client.get_logger().info(
                    'Result of center_estimation: [{}]'.format(response.center_point))
                    # 'Result of center_estimation: [%f]' % (response.center_point))

            break

    reel_center_3d_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()