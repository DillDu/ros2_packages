
import sys
import rclpy
from rclpy.node import Node
from reel_detection_interfaces.srv import DetectReelCenter3D

import cv2
from cv_bridge import CvBridge
from reel_detection_srv.main_code import transformation as tr

class ReelCenterEstimationClient(Node):

    def __init__(self):
        super().__init__('reel_center_estimation_client')
        self.cli = self.create_client(DetectReelCenter3D, 'detect_reel_center_3d')       
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = DetectReelCenter3D.Request()                                   

    def send_request(self):
        bridge = CvBridge()
        img_path1 = sys.argv[1]
        img_path2 = sys.argv[2]
        pose_path1 = sys.argv[3]
        pose_path2 = sys.argv[4]
        
        img_msgs = [bridge.cv2_to_imgmsg(cv2.imread(img_path1), "bgr8"),
                    bridge.cv2_to_imgmsg(cv2.imread(img_path2), "bgr8")]
        rot1, pos1 = tr.read_transform_data(pose_path1)
        rot2, pos2 = tr.read_transform_data(pose_path2)
    
        self.req.img_msgs = img_msgs
        self.req.pos1 = pos1.tolist()
        self.req.pos2 = pos2.tolist()
        self.req.rot1 = rot1.tolist()
        self.req.rot2 = rot2.tolist()
        
        self.future = self.cli.call_async(self.req)


def main(args=None):
    rclpy.init(args=args)

    reel_center_estimation_client = ReelCenterEstimationClient()
    reel_center_estimation_client.send_request()

    while rclpy.ok():
        rclpy.spin_once(reel_center_estimation_client)
        if reel_center_estimation_client.future.done():
            try:
                response = reel_center_estimation_client.future.result()
            except Exception as e:
                reel_center_estimation_client.get_logger().info(
                    'Service call failed %r' % (e,))
            else:
                reel_center_estimation_client.get_logger().info(
                    'Result of center_estimation: [{}]'.format(response.center_point))
                    # 'Result of center_estimation: [%f]' % (response.center_point))
                bridge = CvBridge()
                imgs = []
                msgs = response.result_img_msgs
                for i in range(len(msgs)):
                    img = bridge.imgmsg_to_cv2(msgs[i])
                    imgs.append(img)
                    cv2.imshow('img:{}'.format(i),img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            break

    reel_center_estimation_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()