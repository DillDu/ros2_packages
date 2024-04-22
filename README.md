# ros2_pacakges
init:
source /opt/ros/foxy/setup.bash
source install/setup.bash

force build:
colcon build --cmake-force-configure --packages-select my_package1 my_package2

build commands:
source install/setup.bash 
colcon build --packages-select reel_detection_interfaces 
colcon build --packages-select reel_detection_srv 

ros2 run reel_detection_srv reel_center_estimation_service

ros2 run reel_detection_srv reel_center_estimation_client ~/Documents/20231201/reel_0_1280/reel_0_1280_5_img.png ~/Documents/20231201/reel_0_1280/reel_0_1280_7_img.png ~/Documents/20231201/reel_0_1280/reel_0_1280_robot_pose.txt

ros2 run reel_detection_srv reel_center_estimation_client ~/Documents/20231201/reel_0_1280/reel_0_1280_6_img.png ~/Documents/20231201/reel_0_1280/reel_0_1280_8_img.png ~/Documents/20231201/reel_0_1280/reel_0_1280_robot_pose.txt

ros2 run reel_detection_srv reel_center_estimation_client ~/Documents/20231201/reel_1_1280/reel_1_1280_5_img.png ~/Documents/20231201/reel_1_1280/reel_1_1280_7_img.png ~/Documents/20231201/reel_1_1280/reel_1_1280_robot_pose.txt

ros2 run reel_detection_srv reel_center_estimation_client ~/Documents/20231201/reel_2_1280/reel_2_1280_5_img.png ~/Documents/20231201/reel_2_1280/reel_2_1280_7_img.png ~/Documents/20231201/reel_2_1280/reel_2_1280_robot_pose.txt

ros2 run reel_detection_srv reel_center_estimation_client ~/Documents/20231201/reel_3_1280/reel_3_1280_5_img.png ~/Documents/20231201/reel_3_1280/reel_3_1280_7_img.png ~/Documents/20231201/reel_3_1280/reel_3_1280_robot_pose.txt