from setuptools import setup

package_name = 'reel_detection_srv'
submodules = 'reel_detection_srv/main_code'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name, submodules],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Du Xiaohe',
    maintainer_email='dill.dxh@gmail.com',
    description='ROS2 service for ellipse fitting',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 'service = reel_detection_srv.service_function:main',
            # 'client = reel_detection_srv.client_function:main',
            'reel_center_estimation_client = reel_detection_srv.reel_center_estimation_client:main',
            'reel_center_estimation_service = reel_detection_srv.reel_center_estimation_service:main',
        ],
    },
)
