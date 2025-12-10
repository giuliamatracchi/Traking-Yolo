from syndatatoolbox_api.environment import Environment
import numpy as np
import cv2
import time


if __name__ == '__main__':
    # UE5 Image
    setup = {'show': 'None', 'image_folder_path': 'None', 'mask_folder_path': 'None', 'mask_colorized': 'RGB',
             'format_output_mask': '.png', 'segmentation_video_path': 'None', 'bounding_box_file_path': 'None',
             'bounding_box_print_output': 'True', 'video_path': 'None', 'render': 'None', 'max_depth': 200.}
    action_manager = "CoordinateActionManager(CoordinateActionManagerSDT)"
    action_type = "MOVETO"

    sensor_list = ["GPS(GPSSDT)", "RGBCamera(CameraSDT)", "DepthCamera(DepthCameraSDT)",
                   "GPS(GPSSDTAMBULANCE)", "SegmentationCamera(SegmentationCameraSDT)", "Lidar(LidarSDT)"]

    ue_env = Environment(port=9734, address='192.168.1.95', setup=setup)
    ue_to_meter = 100.
    action = [-960.0,
              -11440.0,
              0.,
              0.,
              0.,
              0.,
              1.]
    
    print(ue_env.sensor_set)

    while True:
        _, observations = ue_env.env_step({action_manager: {action_type: action}}, sensor_list)
        bgr_image = observations[1]  # (128, 128, 3)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        depth_image = observations[2][250]
        print(depth_image)
        segmentation_image = observations[4]
        lidar = observations[5]
        # print(lidar)
        cv2.imshow("Segmentation", segmentation_image)
        cv2.imshow("obs", rgb_image)
        cv2.waitKey(1)
        time.sleep(0.5)