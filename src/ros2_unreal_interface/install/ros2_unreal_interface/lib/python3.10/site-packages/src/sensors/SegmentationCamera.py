from .sensor import Sensor
import numpy as np
import math
import cv2
import os
import rclpy
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError

class SegmentationCamera(Sensor):
  
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

        ns_sensor = f"/{str(self._env_topic).lstrip('/')}/{str(self._sensor_type).strip()}"
        ns_specific = f"{ns_sensor}/{str(self._specific_name).strip()}"
        base_topic = ns_specific
        self._base_topic = base_topic
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

        # verifica forma: accetta 2D o 3D con canale 1
        if not isinstance(data, np.ndarray):
            return
        mask = data
        if mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask[:, :, 0]
        if mask.ndim != 2:
            return

        # non accettare float (niente normalizzazione): solo interi (Domanda: possono arrivare float? Tengo o tolgo?)
        if np.issubdtype(mask.dtype, np.floating):
            return  #rifiuta immagine float per evitare normalizzazioni implicite

        # Scegli dtype ed encoding in base al range: se classe <= 255 mono8 altrimenti mono 16
        encoding = None
        if mask.dtype == np.uint8:
            encoding = "mono8"
        elif mask.dtype == np.uint16:
            encoding = "mono16"
        else:
            # cast per altri interi (es. int32/uint32)
            min_v = int(mask.min()) if mask.size > 0 else 0
            max_v = int(mask.max()) if mask.size > 0 else 0
            if min_v < 0:
                return
            if max_v <= 255:
                mask = mask.astype(np.uint8, copy=False)
                encoding = "mono8"
            elif max_v <= 65535:
                mask = mask.astype(np.uint16, copy=False)
                encoding = "mono16"
            else: #ho valori troppo grandi
                return

        try:
            h_img, w_img = mask.shape[:2]
        except (AttributeError,ValueError):
            return

        # aggiorna CameraInfo se mancano width/height o sono 0
        if int(self._obs_settings.get('height', 0) or 0) == 0 or int(self._obs_settings.get('width', 0) or 0) == 0:
            self._obs_settings['height'] = int(h_img)
            self._obs_settings['width'] = int(w_img)
            if 'camera_matrix' not in self._obs_settings or np.array(self._obs_settings['camera_matrix']).size != 9:
                self._obs_settings['camera_matrix'] = np.zeros((3, 3), dtype=np.float64)
            self.__camera_info_msg = self.build_camera_info_msg(self._obs_settings)

        h = Header()
        h.frame_id = self.camera_frame
        h.stamp = self._node.get_clock().now().to_msg()

        # CameraInfo
        if self._pub_info is not None and self.__camera_info_msg is not None:
            self.__camera_info_msg.header = h
            self.__camera_info_msg.width = int(w_img)
            self.__camera_info_msg.height = int(h_img)
            self._pub_info.publish(self.__camera_info_msg)

        # pubblicazione della vera mask 
        if self._publish_raw and self._pub_raw is not None:
            try:
                img_raw = self.bridge.cv2_to_imgmsg(mask, encoding)
            except CvBridgeError:
                return
            img_raw.header = h
            self._pub_raw.publish(img_raw)

        # immagine compressA
        if self._publish_comp and self._pub_comp is not None:
            # Garantire contiguità del buffer
            if not mask.flags.c_contiguous:
                mask_to_enc = np.ascontiguousarray(mask)
            else:
                mask_to_enc = mask
            # Usa PNG per evitare perdite
            ext = ".png"
            # Per mono16, OpenCV si aspetta uint16; per mono8, uint8
            success, buf = cv2.imencode(ext, mask_to_enc)
            if success:
                comp = CompressedImage()
                comp.header = h
                comp.format = "png"
                comp.data = np.array(buf).tobytes()
                self._pub_comp.publish(comp)

      

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

    