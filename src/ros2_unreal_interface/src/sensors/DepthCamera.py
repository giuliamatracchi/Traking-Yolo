from .sensor import Sensor
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from cv_bridge import CvBridge

class DepthCamera(Sensor):
    def __init__(self, node, env_topic, obs_settings, camera_frame: str = 'camera_link', topic_raw: str = "/image_raw", queue_size_raw: int = 1, topic_compressed: str = "/image_raw/compressed", publish_compressed: bool = False, queue_size_compressed: int = 1, topic_compressed_preview: str = "/image_preview/compressed", publish_compressed_preview: bool = False, queue_size_compressed_preview: int = 1, **kwargs):
        super().__init__()
        self.bridge = CvBridge()

        self._node = node
        self._env_topic = env_topic  # nodo dell'ambiente
        self._obs_settings = obs_settings
        self._sensor_type = kwargs.get('sensor_type', 'camera')  # tipo di sensore passato da environment.py
        self._specific_name = kwargs.get('specific_name', 'default')  # "nome" tra parentesi passato da environment.py
        self.camera_frame = camera_frame

     
        self._raw_encoding = str(kwargs.get('raw_encoding', 'mono8')) #sceglie il formato del messaggio - utilizzo 'mono8' perché voglio vedere qualcosa subito in 
        self._raw_scale = float(kwargs.get('raw_scale', 1.0))#fattore di scala applicato solo qunado pubblico 32FC1 (ovvero in metri), di default 1 per mantenere in metri, se metti 1000 va in mm
        

        # Normalizzazione per frame (min-max) quando si pubblica mono8 - essendo TRUR normalizza per frame con min-max del frame stesso
        self._raw_auto_normalize = bool(kwargs.get('raw_auto_normalize', True))

        # Debug per ispezionare i frame depth, analizza dimensioni, attaulmente disabilitato
        self._debug_depth = bool(kwargs.get('debug_depth', False))
        self._debug_depth_frames = int(kwargs.get('debug_depth_frames', 5))
        self._debug_depth_count = 0

    
        # Pubblicazione opzionale di uno stream tecnico 32FC1 su topic separato
        self._publish_float_raw = bool(kwargs.get('publish_float_raw', False))
        self._float_raw_topic = str(kwargs.get('float_raw_topic', 'image_raw_32fc1')) #nome del suffisso del topic, viene concatenato al name_space del sensore
        self._pub_float_raw = None

        self._expects_data = True #indica al ciclo di pubblicazione in Environment.publish_loop che questo sensore necessita di dati in ingresso (depth) per poter pubblicare.


        # Helper per comporre topic senza slash iniziale, rispettando il namespace del nodo
        def _join_topic(*parts: str) -> str:
            cleaned = [str(p).strip().strip('/') for p in parts if p is not None and str(p).strip() != ""]
            return "/".join(cleaned)

       
        base_topic = _join_topic(self._env_topic, self._sensor_type, self._specific_name)#costruisco il prefisso comune dei topic del sensore

        # Publisher per CameraInfo
        self._pub_info = self._node.create_publisher(CameraInfo, _join_topic(base_topic, "camera_info"), 1)

        # Publisher per immagine depth raw
        raw_topic = _join_topic(base_topic, str(topic_raw))
        self._pub_raw = self._node.create_publisher(Image, raw_topic, queue_size_raw)

      

        # Publisher per eventuale stream metrico in 32FC1
        if self._publish_float_raw:
            float_topic = _join_topic(base_topic, self._float_raw_topic)
            self._pub_float_raw = self._node.create_publisher(Image, float_topic, 1)
    
       
        self.__camera_info_msg = self.build_camera_info_msg(self._obs_settings)

    
    def change_settings(self):
        pass

    
    def publish_observation(self, data):
        if data is None:
            return

        try:
            depth = np.asarray(data, dtype=np.float32)
        except (TypeError, ValueError):
            return

        # Rimuovi dimensioni di size 1, poi prova reshape 1D->2D se conosco H/W
        depth = np.squeeze(depth)
        h0 = int(self._obs_settings.get('height', 0) or 0)
        w0 = int(self._obs_settings.get('width', 0) or 0)
        if depth.ndim == 1 and h0 > 0 and w0 > 0 and depth.size == h0 * w0:
            depth = depth.reshape(h0, w0)

        # Se 3D, usa il primo canale; se non 2D alla fine, abort
        if depth.ndim == 3:
            depth = depth[..., 0]
        if depth.ndim != 2:
            return

        # Dimensioni immagine; se non ottenibili, non pubblica
        try:
            h_img, w_img = depth.shape[:2]
        except Exception:
            return

        # Inizializza CameraInfo solo al primo frame se H/W non sono noti
        if int(self._obs_settings.get('height', 0) or 0) == 0 or int(self._obs_settings.get('width', 0) or 0) == 0: # controlla se sono 0 o mancanti
            #sono mancanti-->imposta con le dim effettive dell'immagine
            self._obs_settings['height'] = int(h_img)
            self._obs_settings['width'] = int(w_img)
            #se la camera_matrix manca o non è 3×3, crea una matrice 3×3 di zeri come fallback
            if 'camera_matrix' not in self._obs_settings or np.array(self._obs_settings['camera_matrix']).size != 9:
                self._obs_settings['camera_matrix'] = np.zeros((3, 3), dtype=np.float64)
            self.__camera_info_msg = self.build_camera_info_msg(self._obs_settings)

        # Header condiviso
        h = Header()
        h.frame_id = self.camera_frame
        h.stamp = self._node.get_clock().now().to_msg()

        # Pubblica CameraInfo
        if self._pub_info is not None and self.__camera_info_msg is not None:
            self.__camera_info_msg.header = h
            self._pub_info.publish(self.__camera_info_msg)

        # pulizia valori + scala metrica per normalizzazione in mono8 o pubblicazione in 32FC1 (m)
        depth_proc = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        max_d = float(self._obs_settings.get('max_depth', 200.0) or 200.0)
        if max_d <= 0:
            max_d = 1.0

        # Pubblica RAW in base a _raw_encoding
        if self._pub_raw is not None:
            if str(self._raw_encoding).upper() in ("MONO8", "8UC1"): #Se _raw_auto_normalize è True: calcola min e max del frame
               
                if self._raw_auto_normalize:
                    dmin = float(np.min(depth_proc))
                    dmax = float(np.max(depth_proc))
                    rng = dmax - dmin
                    if rng > 1e-6:
                        depth_norm = (depth_proc - dmin) / rng
                    else:
               
                        depth_norm = depth_proc / max_d
                else:
                    depth_norm = depth_proc / max_d

                # Assicura valori nell'intervallo [0,1]
                depth_norm = np.clip(depth_norm, 0.0, 1.0)
                img8 = (depth_norm * 255.0).astype(np.uint8) #converte a 0..255 per garantire memoria contigua
                if not img8.flags.c_contiguous:
                    img8 = np.ascontiguousarray(img8)
                img_raw = self.bridge.cv2_to_imgmsg(img8, encoding="mono8")
            #caso 32FC1
            else:
                depth_for_raw = (depth * float(self._raw_scale)).astype(np.float32, copy=False)
                if not depth_for_raw.flags.c_contiguous:
                    depth_for_raw = np.ascontiguousarray(depth_for_raw)
                img_raw = self.bridge.cv2_to_imgmsg(depth_for_raw, encoding="32FC1")
            img_raw.header = h
            self._pub_raw.publish(img_raw)


        # pubblica, su richiesta, un secondo stream “tecnico” della depth in formato 32FC1, metri normalizzati
        if self._publish_float_raw and self._pub_float_raw is not None:
            depth_32f = depth.astype(np.float32, copy=False)
            if not depth_32f.flags.c_contiguous:
                depth_32f = np.ascontiguousarray(depth_32f)
            img_raw32 = self.bridge.cv2_to_imgmsg(depth_32f, encoding="32FC1")
            img_raw32.header = h
            self._pub_float_raw.publish(img_raw32)

  

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

        cam_m_raw = (obs_settings or {}).get('camera_matrix', np.zeros((3, 3), dtype=np.float64))
        cam_arr = np.array(cam_m_raw, dtype=np.float64)
        if cam_arr.size != 9:
            cam_arr = np.zeros((3, 3), dtype=np.float64)
        cam_m = cam_arr.reshape(3, 3)
        msg.k = cam_m.flatten().tolist()
        msg.r = [1., 0., 0., 0., 1., 0., 0., 0., 1.]

        # Proiezione con colonna di traslazione nulla
        P = np.concatenate((cam_m, np.zeros((3, 1), dtype=np.float64)), axis=1).flatten()
        msg.p = P.tolist()
        msg.binning_x = 0
        msg.binning_y = 0
        return msg
