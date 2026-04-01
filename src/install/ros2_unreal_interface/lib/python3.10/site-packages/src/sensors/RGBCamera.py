from .sensor import Sensor
import numpy as np
import math
import cv2
import os
import rclpy
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError


class RGBCamera(Sensor):
  
    def __init__(self, node, env_topic, obs_settings, camera_frame: str = 'camera_link', topic_raw: str = "/image_raw", publish_raw: bool = True, queue_size_raw: int = 1, topic_compressed: str = "/image_raw/compressed", publish_compressed: bool = False, queue_size_compressed: int = 1, input_bgr: bool = True, **kwargs):
        super().__init__()
        self.bridge = CvBridge()
    
        self._node = node
        self._env_topic = env_topic #nodo dell'ambiente 
        self._obs_settings = obs_settings
        self._unreal_settings = kwargs.get('unreal_settings', {})
        self._ue = kwargs.get('environment_ue')
        self._ue_sensor = kwargs.get('ue_sensor')
        self._sensor_type = kwargs.get('sensor_type', 'camera') #tipo di sensore passato da nevironment.py
        self._specific_name = kwargs.get('specific_name', 'default') #"nome" tra parentesi passato da environment.py
        self.camera_frame = camera_frame
        
        self._input_bgr = bool(input_bgr)
  
        self._publish_raw = bool(publish_raw)
        self._publish_comp = bool(publish_compressed)
        
        self._expects_data = True

        # Build topic namespace: /<env_topic>/<sensor_type>/<specific_name>
        ns_sensor = f"/{str(self._env_topic).lstrip('/')}/{str(self._sensor_type).strip()}"
        ns_specific = f"{ns_sensor}/{str(self._specific_name).strip()}"
        base_topic = ns_specific
        self._pub_info = self._node.create_publisher(CameraInfo, f"{base_topic}/camera_info", 1)

        raw_topic = f"{base_topic}/{str(topic_raw).lstrip('/')}" #normalizzazione del suffisso 
        self._pub_raw = self._node.create_publisher(Image, raw_topic, queue_size_raw)

        comp_topic = f"{base_topic}/{str(topic_compressed).lstrip('/')}"
        # Rispetta il flag publish_compressed: crea il publisher solo se abilitato
        self._pub_comp = self._node.create_publisher(CompressedImage, comp_topic, queue_size_compressed) if self._publish_comp else None

        self.__camera_info_msg = self.build_camera_info_msg(self._obs_settings)
        

    def change_settings(self):
        pass


    def publish_observation(self, data):
        if data is None:
            return

       
        if isinstance(data, np.ndarray) and data.dtype != np.uint8:
            img_fixed = np.clip(data * 255.0, 0, 255).astype(np.uint8)
        else:
            img_fixed = data
        try:
            h_img, w_img = img_fixed.shape[:2]
        except Exception:
            return

        if int(self._obs_settings.get('height', 0) or 0) == 0 or int(self._obs_settings.get('width', 0) or 0) == 0:
            self._obs_settings['height'] = int(h_img)
            self._obs_settings['width'] = int(w_img)
            if 'camera_matrix' not in self._obs_settings or np.array(self._obs_settings['camera_matrix']).size != 9:
                self._obs_settings['camera_matrix'] = np.zeros((3, 3), dtype=np.float64)
            self.__camera_info_msg = self.build_camera_info_msg(self._obs_settings)

   
        h = Header()
        h.frame_id = self.camera_frame
        h.stamp = self._node.get_clock().now().to_msg()


        if self._pub_info is not None and self.__camera_info_msg is not None:
            self.__camera_info_msg.header = h
            self._pub_info.publish(self.__camera_info_msg)

        if self._publish_raw and self._pub_raw is not None:
            img_rgb = img_fixed
            img_raw = self.bridge.cv2_to_imgmsg(img_rgb, "rgb8")
            img_raw.header = h
            self._pub_raw.publish(img_raw)

      
        if self._publish_comp and self._pub_comp is not None:
            img_for_comp = img_fixed
           
            if isinstance(img_for_comp, np.ndarray):
                if img_for_comp.dtype != np.uint8:
                    img_for_comp = np.clip(img_for_comp, 0, 255).astype(np.uint8)
                if img_for_comp.ndim == 2 or (img_for_comp.ndim == 3 and img_for_comp.shape[2] == 1):
                    img_for_comp = cv2.cvtColor(img_for_comp, cv2.COLOR_GRAY2BGR)
                elif img_for_comp.ndim == 3 and img_for_comp.shape[2] == 4:
                  
                    try:
                        img_for_comp = cv2.cvtColor(img_for_comp, cv2.COLOR_RGBA2BGR)
                    except Exception:
                        img_for_comp = img_for_comp[:, :, :3]
                elif img_for_comp.ndim == 3 and img_for_comp.shape[2] == 3:
                
                    try:
                        img_for_comp = cv2.cvtColor(img_for_comp, cv2.COLOR_RGB2BGR)
                    except Exception:
                        pass
                if not img_for_comp.flags.c_contiguous:
                    img_for_comp = np.ascontiguousarray(img_for_comp)

            img_compressed = self.bridge.cv2_to_compressed_imgmsg(img_for_comp)
            img_compressed.header = h
            self._pub_comp.publish(img_compressed)




    def build_camera_info_msg(self, obs_settings):
        h = Header()
       
        h.stamp = self._node.get_clock().now().to_msg()
        h.frame_id = self.camera_frame

        msg = CameraInfo()
        msg.header = h
        msg.height = int(obs_settings.get('height', 0) or 0)
        msg.width = int(obs_settings.get('width', 0) or 0)
        msg.distortion_model = "plumb_bob"
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]

        cam_m_raw = (obs_settings or {}).get(
            'camera_matrix', np.zeros((3, 3), dtype=np.float64)
        )
        cam_arr = np.array(cam_m_raw, dtype=np.float64)
        if cam_arr.size != 9:
            cam_arr = np.zeros((3, 3), dtype=np.float64)
        cam_m = cam_arr.reshape(3, 3)
        msg.k = cam_m.flatten().tolist()
        msg.r = [1., 0., 0., 0., 1., 0., 0., 0., 1.]
        P = np.concatenate((cam_m, np.zeros((3, 1), dtype=np.float64)), axis=1).flatten()
        msg.p = P.tolist()
        msg.binning_x = 0
        msg.binning_y = 0
        return msg
