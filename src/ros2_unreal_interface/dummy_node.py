import numpy as np
import cv2
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from std_msgs.msg import Header
from cv_bridge import CvBridge
from syndatatoolbox_api.environment import Environment


class dummy_node(Node):
    def __init__(self):
        super().__init__('dummy_node')

        
        self.player_topic = '/camera'
        self.camera_frame = 'camera_link'
        self.rate_hz = 2.0

        setup = {'show': 'None', 'image_folder_path': 'None', 'mask_folder_path': 'None', 'mask_colorized': 'RGB',
             'format_output_mask': '.png', 'segmentation_video_path': 'None', 'bounding_box_file_path': 'None',
             'bounding_box_print_output': 'True', 'video_path': 'None', 'render': 'None', 'max_depth': 200.}
    
        self.action_manager = "CoordinateActionManager(CoordinateActionManagerSDT)"
        self.action_type = "MOVETO"
        self.sensor_list = [
            "GPS(GPSSDT)", "RGBCamera(CameraSDT)", "DepthCamera(DepthCameraSDT)",
            "GPS(GPSSDTAMBULANCE)"
        ]
        self.action = [-960.0, -11440.0, 0.0, 0.0, 0.0, 0.0, 1.0]

        
        self.ue_env = Environment(port=9734, address='192.168.1.95', setup=setup)
        self.get_logger().info(str(self.ue_env.sensor_set))

        
        self._bridge = CvBridge()
        self._pub_info = self.create_publisher(CameraInfo, f'{self.player_topic}/camera/camera_info', 1)
        self._pub_raw = self.create_publisher(Image, f'{self.player_topic}/camera/image_raw', 1)
        self._pub_comp = self.create_publisher(CompressedImage, f'{self.player_topic}/camera/image_raw/compressed', 1)

        
        self.obs_settings = {
            'height': 0,
            'width': 0,
            'camera_matrix': np.zeros((3, 3), dtype=np.float32)
        }
        self._camera_info_msg = self.build_camera_info_msg(self.obs_settings)

       
        def _step():
            try:
                _, observations = self.ue_env.env_step(
                    {self.action_manager: {self.action_type: self.action}},
                    self.sensor_list
                )
                bgr_image = observations[1 ]
                if bgr_image is None:
                    self.get_logger().warning("osservazione nulla")
                    return

             
                if self.obs_settings['height'] == 0 or self.obs_settings['width'] == 0:
                    h, w = bgr_image.shape[:2]
                    self.obs_settings['height'] = int(h)
                    self.obs_settings['width'] = int(w)
                    self.obs_settings['camera_matrix'] = np.zeros((3, 3), dtype=np.float32)
                    self._camera_info_msg = self.build_camera_info_msg(self.obs_settings)

                rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

                self.publish_observation(rgb_image)

            except Exception as e:
                self.get_logger().error(f'Errore ciclo: {e}')

        period = 1.0 / max(self.rate_hz, 0.0001)
        self._timer = self.create_timer(period, _step)

    def publish_observation(self, data):
        img_fixed = data * 255
        img_fixed = img_fixed.astype(dtype=np.uint8)

        h = Header()
        h.frame_id = self.camera_frame
        h.stamp = self.get_clock().now().to_msg()

  
        self._camera_info_msg.header = h
        self._pub_info.publish(self._camera_info_msg)

        img_rgb = cv2.cvtColor(img_fixed, cv2.COLOR_BGR2RGB)
        img_raw = self._bridge.cv2_to_imgmsg(img_rgb, "rgb8")
        img_raw.header = h
        self._pub_raw.publish(img_raw)

        img_compressed = self._bridge.cv2_to_compressed_imgmsg(img_fixed)
        img_compressed.header = h
        self._pub_comp.publish(img_compressed)


    def build_camera_info_msg(self, obs_settings):
        h = Header()
        h.stamp = self.get_clock().now().to_msg()
        h.frame_id = self.camera_frame

        msg = CameraInfo()
        msg.header = h
        msg.height = int(obs_settings.get('height', 0) or 0)
        msg.width = int(obs_settings.get('width', 0) or 0)
        msg.distortion_model = "plumb_bob"
        msg.d = np.zeros((5,), dtype=np.float32).tolist()
        cam_m = obs_settings.get('camera_matrix', np.zeros((3, 3), dtype=np.float32))
        msg.k = cam_m.flatten().tolist()
        msg.r = [1., 0., 0., 0., 1., 0., 0., 0., 1.]
        P = np.concatenate((cam_m, np.zeros((3, 1), dtype=np.float32)), axis=1).flatten()
        msg.p = P.tolist()
        msg.binning_x = 0
        msg.binning_y = 0
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = dummy_node()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
