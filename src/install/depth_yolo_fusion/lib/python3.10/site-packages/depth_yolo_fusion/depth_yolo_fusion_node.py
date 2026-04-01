



import json
import math
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
from rclpy.qos import QoSProfile, ReliabilityPolicy


class DepthYoloFusionNode(Node):

    def __init__(self) -> None:
        super().__init__('depth_yolo_fusion_node')

        # -----------------------------
        # PARAMETRI (con default)
        # -----------------------------
        self.declare_parameter(
            'depth_topic',
            '/environment/DepthCamera/DepthCameraSDT/image_raw'
        )
        self.declare_parameter('detections_topic', 'yolo/detections')
        self.declare_parameter('output_annotated_depth_topic', 'yolo/depth_annotated')
        self.declare_parameter('output_xyz_topic', 'yolo/detections_xyz')
        self.declare_parameter('max_sync_dt_sec', 0.1)   # non usato ora, ma lasciato
        self.declare_parameter('viz_auto_scale', True)
        self.declare_parameter('viz_max_depth_m', 200.0)
        
        #nel setup attaule image_raw è 32FC1 ma i valori reali sono in centimetri --< convertiam subito in metri per avere unità SI coerenti in tutta la pipeline
        self.declare_parameter('depth_scale', 0.01)

        self.depth_topic = self.get_parameter(
            'depth_topic'
        ).get_parameter_value().string_value
        self.detections_topic = self.get_parameter(
            'detections_topic'
        ).get_parameter_value().string_value
        self.output_annotated_depth_topic = self.get_parameter(
            'output_annotated_depth_topic'
        ).get_parameter_value().string_value
        self.output_xyz_topic = self.get_parameter(
            'output_xyz_topic'
        ).get_parameter_value().string_value
        self.max_sync_dt_sec = float(
            self.get_parameter('max_sync_dt_sec').get_parameter_value().double_value
        )
        self.viz_auto_scale = bool(
            self.get_parameter('viz_auto_scale').get_parameter_value().bool_value
        )
        self.viz_max_depth_m = float(
            self.get_parameter('viz_max_depth_m').get_parameter_value().double_value
        )
        self.depth_scale = float(
            self.get_parameter('depth_scale').get_parameter_value().double_value
        )

        self.get_logger().info(f"Depth scale: {self.depth_scale} [m/unità]")
        self.get_logger().info(f"Depth topic: {self.depth_topic}")
        self.get_logger().info(f"Detections topic: {self.detections_topic}")

        self.bridge = CvBridge()

        # Coda delle detection YOLO (timestamp → lista di detezioni)
        self._det_queue: Deque[Tuple[float, List[Dict[str, Any]]]] = deque(maxlen=50)

        # -----------------------------
        # SUBSCRIBERS (QoS best-effort per Unreal)
        # -----------------------------
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT

        self.sub_depth = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            qos,
        )

        self.sub_det = self.create_subscription(
            String,
            self.detections_topic,
            self.detections_callback,
            10,
        )

        # -----------------------------
        # PUBLISHERS
        # -----------------------------
        self.pub_annotated = self.create_publisher(
            Image, self.output_annotated_depth_topic, 10
        )
        self.pub_xyz = self.create_publisher(
            String, self.output_xyz_topic, 10
        )

    # ------------------------------------------------------------------
    def detections_callback(self, msg: String):
        """Riceve il JSON YOLO (con eventuale track_id) e lo mette in coda."""
        try:
            data = json.loads(msg.data)
        except Exception:
            return

        stamp = float(data["header"]["stamp"])
        dets = data.get("detections", [])
        # dets è una lista di dict, ognuno può contenere:
        # class_id, class_name, confidence, box_xyxy, (opzionale) track_id
        self._det_queue.append((stamp, dets))

    # ------------------------------------------------------------------
    def _get_last_detections(self) -> List[Dict[str, Any]]:
        """Ritorna SEMPRE l'ultima lista di detection disponibile."""
        if not self._det_queue:
            return []
        return self._det_queue[-1][1]

    # ------------------------------------------------------------------
    def depth_callback(self, msg: Image):

        # Converti depth con passthrough
        try:
            depth_np = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError:
            return

        # CONVERSIONE A float32 e applicazione scala (cm → m nel nostro caso)
        depth_np = depth_np.astype(np.float32) * self.depth_scale



        depth_np = np.nan_to_num(
            depth_np,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # timestamp frame depth (lo uso solo per pubblicare negli XYZ)
        t_depth = (
            msg.header.stamp.sec
            + msg.header.stamp.nanosec * 1e-9
        )

        # Usa sempre l’ultima lista di detection disponibile (con eventuale track_id)
        dets = self._get_last_detections()

        # Processa: normalizza per visualizzazione + box
        annotated_bgr, det_xyz = self._process(depth_np, dets)

        # Pubblica immagine annotata (bgr8!)
        try:
            img_msg = self.bridge.cv2_to_imgmsg(annotated_bgr, encoding="bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridgeError in cv2_to_imgmsg: {e}")
            return

        img_msg.header = msg.header
        self.pub_annotated.publish(img_msg)

        # Pubblica le coordinate XYZ, mantenendo anche il track_id se presente
        out = {
            "header": {
                "stamp": t_depth,
                "frame_id": msg.header.frame_id,
            },
            "detections_xyz": det_xyz,
        }
        m = String()
        m.data = json.dumps(out)
        self.pub_xyz.publish(m)

    # ------------------------------------------------------------------
    def _process(
        self,
        depth: np.ndarray,
        detections: List[Dict[str, Any]],
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Crea immagine visualizzabile da depth + box YOLO (+ track_id)."""

        # Normalizza depth SOLO per visualizzazione [0,255]
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)

        # depth_norm è float32 → converti a uint8
        depth_u8 = depth_norm.astype(np.uint8)

        # Convertila in BGR8 per poter disegnare
        depth_vis = cv2.cvtColor(depth_u8, cv2.COLOR_GRAY2BGR)

        h, w = depth.shape
        out_xyz: List[Dict[str, Any]] = []

        for det in detections:
            box = det.get("box_xyxy")
            if not box or len(box) != 4:
                continue

            x1, y1, x2, y2 = map(float, box)

            x1i = max(0, min(int(x1), w - 1))
            y1i = max(0, min(int(y1), h - 1))
            x2i = max(0, min(int(x2), w - 1))
            y2i = max(0, min(int(y2), h - 1))

            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            cx = max(0, min(cx, w - 1))
            cy = max(0, min(cy, h - 1))

            z = float(depth[cy, cx])
            if not math.isfinite(z) or z <= 0:
                z = None

            # >>> TRACKING: recupero track_id se presente <<<
            track_id = det.get("track_id", None)   # <<< MODIFICATO

            # Disegno box
            cv2.rectangle(
                depth_vis,
                (x1i, y1i),
                (x2i, y2i),
                (0, 255, 0),
                2
            )

            # Etichetta: nome classe + (id) + distanza
            label_parts = []

            if det.get("class_name") is not None:
                label_parts.append(str(det["class_name"]))

            if track_id is not None:
                label_parts.append(f"ID:{track_id}")  # <<< MODIFICATO

            if z is not None:
                label_parts.append(f"{z:.2f}m")

            label = " ".join(label_parts) if label_parts else "?"

            cv2.putText(
                depth_vis,
                label,
                (x1i, max(0, y1i - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

            out_xyz.append(
                {
                    "class_id": det.get("class_id"),
                    "class_name": det.get("class_name"),
                    "confidence": det.get("confidence"),
                    "box_xyxy": [x1, y1, x2, y2],
                    "center_xy": [cx, cy],
                    "z_m": z,
                    "track_id": track_id,          # <<< MODIFICATO
                }
            )

        return depth_vis, out_xyz


def main(args=None):
    rclpy.init(args=args)
    node = DepthYoloFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()



