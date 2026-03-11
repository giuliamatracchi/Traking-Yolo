import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2  # usato per la conversione immagine (bridge)
import numpy as np
import json
import time
import os
import yaml
from typing import List, Dict, Any, Tuple
from ultralytics import YOLO
from rcl_interfaces.msg import ParameterDescriptor, ParameterType




class YoloDetectorNode(Node):

    def __init__(self) -> None:
        super().__init__("YoloDetector")

        # ----------------------------
        # Lettura config YAML (topic + flag tracking)
        # ----------------------------
        config_path = os.path.join(os.path.dirname(__file__), "yolo_detector_node.yaml")
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
        except Exception as e:
            raise RuntimeError(
                f"Impossibile leggere il file YAML: {config_path} ({e})"
            )

        input_topic_from_yaml = str(config.get("input_topic", "")).strip()
        if not input_topic_from_yaml:
            raise RuntimeError(
                f"Chiave 'input_topic' mancante o vuota in {config_path}"
            )
        self.input_topic = input_topic_from_yaml

        # Flag di tracking da YAML
        track_cfg = config.get("track", False) #prende dal file yaml il valore di truck in booleano; Se self.track_enabled è True, la callback usa _infer_ultralytics_track(...). Se False, usa _infer_ultralytics(...).
        self.track_enabled = bool(track_cfg)
        self.get_logger().info(
            f"YOLO tracking: {'ON' if self.track_enabled else 'OFF'} (track={track_cfg})"
        )

        # ----------------------------
        # Parametri ROS2
        # ----------------------------
        self.declare_parameter("output_image_topic", "yolo/annotated_image")
        self.declare_parameter("output_detections_topic", "yolo/detections")
        self.declare_parameter("model_path", "yolov8x.pt")
        self.declare_parameter("conf_threshold", 0.25)
        self.declare_parameter("iou_threshold", 0.45)
        self.declare_parameter("class_filter",[0, 2],ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER_ARRAY))


        # opzionale: file di configurazione tracker YOLO (bytetrack.yaml, botsort.yaml, ...)
        self.declare_parameter("tracker_cfg", "")

        def _get_str(name: str) -> str:
            return self.get_parameter(name).get_parameter_value().string_value

        def _get_float(name: str) -> float:
            return self.get_parameter(name).get_parameter_value().double_value

        self.output_image_topic = _get_str("output_image_topic") #topic immagine annotata
        self.output_detections_topic = _get_str("output_detections_topic") #topic su cui pubblico le detezioni in Json
        self.model_path = _get_str("model_path") #percorso ai pesi Ultralytics
        self.conf_threshold = _get_float("conf_threshold") #soglia confidenza minima per tenere una detection
        self.iou_threshold = _get_float("iou_threshold")  #soglia sovrapposizioni bounding box
        self.tracker_cfg = _get_str("tracker_cfg")  #fil YAML del traker 

        # legge il parametro class_filter come array di interi e lo converte in lista; lo salva in self.class_filter
        cf_param = self.get_parameter("class_filter").get_parameter_value()
        arr = list(cf_param.integer_array_value)
        self.class_filter = list(arr) if arr else []

        self.get_logger().info(
            "YoloDetectorNode inizializzato con:\n"
            f"  input_topic: {self.input_topic}\n"
            f"  output_image_topic: {self.output_image_topic}\n"
            f"  output_detections_topic: {self.output_detections_topic}\n"
            f"  model_path: {self.model_path}\n"
            f"  conf_threshold={self.conf_threshold} | "
            f"iou_threshold={self.iou_threshold} | class_filter={self.class_filter}\n"
            f"  tracker_cfg: {self.tracker_cfg if self.tracker_cfg else '(default YOLO)'}"
        )

        # ----------------------------
        # YOLO model (Ultralytics)
        # ----------------------------
        self.bridge = CvBridge() #crea il bridge ROS-OpenCV per converire tra sesnor_msgs/Image e immagini cv2
        self.model, self.class_names = self._load_model(self.model_path) #carica il modello Ultralytics YOLO dal percorso model_path e ricava la mappa ID→nome classe

        # ----------------------------
        # ROS2 I/O
        # ----------------------------
        qos_sensor = QoSProfile( #profilo per lo stream camera
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.sub = self.create_subscription( #sottoscrive il topic self.input_topic
            Image, self.input_topic, self.image_callback, qos_sensor
        )
        self.pub_img = self.create_publisher(Image, self.output_image_topic, 10) #publisher immagine
        self.pub_det = self.create_publisher(String, self.output_detections_topic, 10) #publisher detection

        # Watchdog
        self._last_frame_t = None  #time stampt ultimo frame ricevuto
        self._no_frame_warned = False
        self._timer = self.create_timer(5.0, self._watchdog) #ogni 5s chiama watchdog che avvisa se non è mai arrivato un frame oppure se sono passati >= 5s dall'ultimo frame

    # ----------------------------
    # Caricamento modello
    # ----------------------------
    def _load_model(self, model_path: str) -> Tuple[Any, Dict[int, str]]:
 
        # importa YOLO da Ultralytics e istanzia il modello con model_path
        model = YOLO(model_path)
        names = getattr(getattr(model, "model", None), "names", getattr(model, "names", {}))

        if isinstance(names, dict): #estrae i nomi delle classi: legge l'attributo names opppure model.names e lo normalizza in un dict
            class_names = {int(k): v for k, v in names.items()} #se names è già un dict, converte le chiavi a int
        else:
            class_names = {i: n for i, n in enumerate(list(names or []))} #se names è una lista/tuple, crea {indice: nome}.

        self.get_logger().info(f"Loaded Ultralytics YOLO from {model_path}")
        return model, class_names #tupla in cui model è l’oggetto Ultralytics YOLO pronto per predict/track, class_names è la mappa ID→nome classe usata per etichettare le detection.

    # ----------------------------
    # Callback immagine
    # ----------------------------
    def image_callback(self, msg: Image) -> None: #viene chiamato ad ogni sensor_msgs/Image su self.input_topic
        self._last_frame_t = time.time() #aggiorna per segnalare ch eun frame è arrivato
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8") #converte il messaggio ros in un np.darray bgr 

        if self.track_enabled: #capisce se fare detection o+ detection +  tracking 
            dets, annotated = self._infer_ultralytics_track(frame)
        else:
            dets, annotated = self._infer_ultralytics(frame)

        #pubblica immahgine: riconverte in annoteded in sensor_smgs/Image, copia l'haeder originale e pubblica su self.output_image_topic.
        out_img = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        out_img.header = msg.header
        self.pub_img.publish(out_img)

        det_msg = String()
        det_msg.data = json.dumps( #crea un std_msgs/string con json
            {
                "header": {
                    "stamp": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9, 
                    "frame_id": msg.header.frame_id,
                },
                "detections": dets,
            },
            ensure_ascii=False,
        )
        self.pub_det.publish(det_msg)
        #la lista prodotta dall’inferenza (tipicamente class_id, class_name, confidence, box_xyxy, e in tracking anche track_id

    # ----------------------------
    # SOLO DETECTION (model.predict)
    # ----------------------------
    def _infer_ultralytics(
        self, frame_bgr: np.ndarray
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        if not results:
            return [], frame_bgr

        r = results[0]
        annotated = r.plot()  # disegna box e label standard YOLO

        dets: List[Dict[str, Any]] = []
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return dets, annotated

        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()

        for i in range(len(cls)):
            cid = int(cls[i])

            if self.class_filter and cid not in self.class_filter:
                continue

            dets.append(
                {
                    "class_id": cid,
                    "class_name": self.class_names.get(cid, str(cid)),
                    "confidence": float(conf[i]),
                    "box_xyxy": [float(v) for v in xyxy[i].tolist()],
                }
            )

        return dets, annotated

    # ----------------------------
    # DETECTION + TRACKING (model.track)
    # ----------------------------
    def _infer_ultralytics_track(
        self, frame_bgr: np.ndarray
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Usa esclusivamente la funzione track() di YOLOv8, come da documentazione Ultralytics.
        Gli ID di tracking sono presi da boxes.id.
        """
        track_args = {
            "source": frame_bgr,
            "conf": self.conf_threshold,
            "iou": self.iou_threshold,
            "verbose": False,
            "persist": True,  # mantiene lo stato del tracker fra i frame
        }

        # Se specificato, usa un file tracker YOLO (es. bytetrack.yaml)
        if self.tracker_cfg:
            track_args["tracker"] = self.tracker_cfg

        results = self.model.track(**track_args)

        if not results:
            return [], frame_bgr

        r = results[0]
        # Annotated image con box + ID disegnati da YOLO
        annotated = r.plot()

        dets: List[Dict[str, Any]] = []
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return dets, annotated

        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()

        ids = None
        if getattr(boxes, "id", None) is not None:
            ids = boxes.id.cpu().numpy()

        for i in range(len(cls)):
            cid = int(cls[i])

            if self.class_filter and cid not in self.class_filter:
                continue

            det: Dict[str, Any] = {
                "class_id": cid,
                "class_name": self.class_names.get(cid, str(cid)),
                "confidence": float(conf[i]),
                "box_xyxy": [float(v) for v in xyxy[i].tolist()],
            }

            # ID di tracking fornito direttamente da YOLO
            if ids is not None and i < len(ids) and ids[i] is not None:
                try:
                    det["track_id"] = int(ids[i])
                except Exception:
                    pass

            dets.append(det)

        return dets, annotated

    # ----------------------------
    # Watchdog
    # ----------------------------
    def _watchdog(self) -> None:
        now = time.time()
        if self._last_frame_t is None:
            self.get_logger().warn(f"Nessun frame ricevuto finora su {self.input_topic}")
        elif (now - self._last_frame_t) > 5.0 and not self._no_frame_warned:
            self._no_frame_warned = True
            self.get_logger().warn(
                f"Nessun frame negli ultimi 5s su {self.input_topic}"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = YoloDetectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()



