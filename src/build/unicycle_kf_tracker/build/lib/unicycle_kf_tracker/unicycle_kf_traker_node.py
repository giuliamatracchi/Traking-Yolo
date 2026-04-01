#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class VehicleParams:
    vehicle_width_m: float = 3.0
    wheelbase_m: float = 1.5


@dataclass
class EKFConfig:
    dt: float = 0.1
    gate_threshold: float = 9.21
    sigma_proc_x: float = 0.10
    sigma_proc_y: float = 0.10
    sigma_proc_psi: float = np.deg2rad(2.0)
    sigma_proc_v: float = 0.75
    sigma_proc_omega: float = np.deg2rad(12.0)
    sigma_meas_x: float = 0.80
    sigma_meas_y: float = 0.80

    def Q(self) -> np.ndarray:
        return np.diag([
            self.sigma_proc_x ** 2,
            self.sigma_proc_y ** 2,
            self.sigma_proc_psi ** 2,
            self.sigma_proc_v ** 2,
            self.sigma_proc_omega ** 2,
        ])

    def R(self) -> np.ndarray:
        return np.diag([
            self.sigma_meas_x ** 2,
            self.sigma_meas_y ** 2,
        ])


class UnicycleEKF:
    """
    Stato:
      x = [X, Y, psi, v, omega]

    Misura:
      z = [X, Y]

    Modello:
      X_{k+1}     = X + v cos(psi) dt
      Y_{k+1}     = Y + v sin(psi) dt
      psi_{k+1}   = psi + omega dt
      v_{k+1}     = v + a dt
      omega_{k+1} = omega
    """

    def __init__(self, config: EKFConfig, x0: np.ndarray, P0: np.ndarray) -> None:
        self.config = config
        self.x = np.asarray(x0, dtype=float).reshape(5)
        self.P = np.asarray(P0, dtype=float).reshape(5, 5)
        self.Q = config.Q()
        self.R = config.R()

        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
        ], dtype=float)

        self.I = np.eye(5, dtype=float)

    def set_dt(self, dt: float) -> None:
        self.config.dt = float(dt)

    def reset(self, x0: np.ndarray, P0: np.ndarray) -> None:
        self.x = np.asarray(x0, dtype=float).reshape(5)
        self.P = np.asarray(P0, dtype=float).reshape(5, 5)

    def predict(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a = float(u[0])
        dt = self.config.dt

        X, Y, psi, v, omega = self.x

        Xn = X + dt * v * math.cos(psi)
        Yn = Y + dt * v * math.sin(psi)
        psin = wrap_angle(psi + dt * omega)
        vn = v + dt * a
        omegan = omega

        self.x = np.array([Xn, Yn, psin, vn, omegan], dtype=float)

        F = np.eye(5, dtype=float)
        F[0, 2] = -dt * v * math.sin(psi)
        F[0, 3] =  dt * math.cos(psi)
        F[1, 2] =  dt * v * math.cos(psi)
        F[1, 3] =  dt * math.sin(psi)
        F[2, 4] =  dt

        self.P = F @ self.P @ F.T + self.Q
        self.x[2] = wrap_angle(self.x[2])

        return self.x.copy(), self.P.copy()

    def innovation(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        z = np.asarray(z, dtype=float).reshape(2)
        z_pred = self.H @ self.x
        y = z - z_pred
        S = self.H @ self.P @ self.H.T + self.R
        return y, S

    def update(self, z: np.ndarray) -> Dict[str, Any]:
        y, S = self.innovation(z)
        Sinv = np.linalg.inv(S)
        nis = float(y.T @ Sinv @ y)

        result: Dict[str, Any] = {
            "accepted": False,
            "nis": nis,
            "innovation": y.copy(),
            "state": self.x.copy(),
            "cov": self.P.copy(),
        }

        if nis > self.config.gate_threshold:
            return result

        K = self.P @ self.H.T @ Sinv
        self.x = self.x + K @ y
        self.x[2] = wrap_angle(self.x[2])

        I_KH = self.I - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        result.update({
            "accepted": True,
            "state": self.x.copy(),
            "cov": self.P.copy(),
        })
        return result

    def step(self, u: np.ndarray, z: Optional[np.ndarray] = None) -> Dict[str, Any]:
        self.predict(u)

        if z is None:
            return {
                "accepted": False,
                "nis": float("nan"),
                "innovation": np.full(2, np.nan),
                "state": self.x.copy(),
                "cov": self.P.copy(),
            }

        return self.update(z)


@dataclass
class TrackMemory:
    ekf: UnicycleEKF
    last_t: float
    last_meas_xy: np.ndarray
    last_meas_heading: Optional[float] = None
    last_meas_speed: float = 0.0
    last_meas_omega: float = 0.0
    class_name: str = ""
    last_box_xyxy: Optional[List[float]] = None


class UnicycleKFTrackerNode(Node):

    def __init__(self) -> None:
        super().__init__("unicycle_kf_tracker")

        self.declare_parameter("input_topic", "yolo/detections_xyz")
        self.declare_parameter("output_topic", "unicycle_kf/tracks")

        self.declare_parameter("image_width_px", 1280)
        self.declare_parameter("horizontal_fov_deg", 90.0)

        self.declare_parameter("vehicle_width_m", 3.0)
        self.declare_parameter("wheelbase_m", 1.5)

        self.declare_parameter("dt_default", 0.1)
        self.declare_parameter("gate_threshold", 9.21)
        self.declare_parameter("sigma_proc_x", 0.10)
        self.declare_parameter("sigma_proc_y", 0.10)
        self.declare_parameter("sigma_proc_psi_deg", 2.0)
        self.declare_parameter("sigma_proc_v", 0.75)
        self.declare_parameter("sigma_proc_omega_deg", 12.0)
        self.declare_parameter("sigma_meas_x", 0.80)
        self.declare_parameter("sigma_meas_y", 0.80)

        self.declare_parameter("track_timeout_sec", 1.0)
        self.declare_parameter("min_dt_sec", 0.03)
        self.declare_parameter("max_dt_sec", 0.50)
        self.declare_parameter("min_valid_depth_m", 0.1)

        self.declare_parameter("allowed_classes", [])

        self.input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        self.output_topic = self.get_parameter("output_topic").get_parameter_value().string_value

        self.image_width_px = int(self.get_parameter("image_width_px").value)
        self.horizontal_fov_deg = float(self.get_parameter("horizontal_fov_deg").value)

        self.vehicle_width_m = float(self.get_parameter("vehicle_width_m").value)
        self.wheelbase_m = float(self.get_parameter("wheelbase_m").value)

        self.dt_default = float(self.get_parameter("dt_default").value)
        self.gate_threshold = float(self.get_parameter("gate_threshold").value)
        self.sigma_proc_x = float(self.get_parameter("sigma_proc_x").value)
        self.sigma_proc_y = float(self.get_parameter("sigma_proc_y").value)
        self.sigma_proc_psi_deg = float(self.get_parameter("sigma_proc_psi_deg").value)
        self.sigma_proc_v = float(self.get_parameter("sigma_proc_v").value)
        self.sigma_proc_omega_deg = float(self.get_parameter("sigma_proc_omega_deg").value)
        self.sigma_meas_x = float(self.get_parameter("sigma_meas_x").value)
        self.sigma_meas_y = float(self.get_parameter("sigma_meas_y").value)

        self.track_timeout_sec = float(self.get_parameter("track_timeout_sec").value)
        self.min_dt_sec = float(self.get_parameter("min_dt_sec").value)
        self.max_dt_sec = float(self.get_parameter("max_dt_sec").value)
        self.min_valid_depth_m = float(self.get_parameter("min_valid_depth_m").value)

        allowed = self.get_parameter("allowed_classes").value
        self.allowed_classes = set(str(x) for x in allowed) if allowed else set()

        hfov_rad = math.radians(self.horizontal_fov_deg)
        self.focal_px = self.image_width_px / (2.0 * math.tan(hfov_rad / 2.0))

        self.vehicle_params = VehicleParams(
            vehicle_width_m=self.vehicle_width_m,
            wheelbase_m=self.wheelbase_m,
        )

        self.ekf_config = EKFConfig(
            dt=self.dt_default,
            gate_threshold=self.gate_threshold,
            sigma_proc_x=self.sigma_proc_x,
            sigma_proc_y=self.sigma_proc_y,
            sigma_proc_psi=math.radians(self.sigma_proc_psi_deg),
            sigma_proc_v=self.sigma_proc_v,
            sigma_proc_omega=math.radians(self.sigma_proc_omega_deg),
            sigma_meas_x=self.sigma_meas_x,
            sigma_meas_y=self.sigma_meas_y,
        )

        self.tracks: Dict[int, TrackMemory] = {}

        self.sub = self.create_subscription(
            String,
            self.input_topic,
            self.detections_callback,
            10,
        )
        self.pub = self.create_publisher(String, self.output_topic, 10)

        self.cleanup_timer = self.create_timer(0.5, self._cleanup_stale_tracks)

        self.get_logger().info(
            "UnicycleKFTrackerNode ready\n"
            f"  input_topic: {self.input_topic}\n"
            f"  output_topic: {self.output_topic}\n"
            f"  image_width_px: {self.image_width_px}\n"
            f"  horizontal_fov_deg: {self.horizontal_fov_deg}\n"
            f"  vehicle_width_m: {self.vehicle_width_m}\n"
            f"  wheelbase_m: {self.wheelbase_m}\n"
            f"  focal_px: {self.focal_px:.2f}"
        )

    def _cleanup_stale_tracks(self) -> None:
        now = time.time()
        stale_ids = [
            tid for tid, mem in self.tracks.items()
            if (now - mem.last_t) > self.track_timeout_sec
        ]
        for tid in stale_ids:
            del self.tracks[tid]

    def _class_allowed(self, class_name: str) -> bool:
        if not self.allowed_classes:
            return True
        return class_name in self.allowed_classes

    def _measurement_from_detection(self, det: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Converte una detection in misura metrica [x, y] nel frame camera:
          x = distanza in avanti [m]
          y = laterale [m]

        Mantiene la stessa convenzione del tracker Ackermann:
        il campo z_m viene trattato come centimetri e convertito in metri.
        """
        z_raw = det.get("z_m", None)
        center_xy = det.get("center_xy", None)
        box_xyxy = det.get("box_xyxy", None)

        if z_raw is None:
            return None

        try:
            z_m = float(z_raw) / 100.0
        except Exception:
            return None

        if not math.isfinite(z_m) or z_m < self.min_valid_depth_m:
            return None

        if center_xy is None or len(center_xy) != 2:
            return None

        try:
            cx = float(center_xy[0])
        except Exception:
            return None

        y_lateral = ((cx - (self.image_width_px / 2.0)) * z_m) / self.focal_px

        # correzione opzionale con larghezza bbox, usando vehicle_width_m = 3.0 m
        if box_xyxy is not None and len(box_xyxy) == 4:
            try:
                x1, _, x2, _ = map(float, box_xyxy)
                bbox_w = max(1.0, abs(x2 - x1))
                z_from_width = (self.focal_px * self.vehicle_params.vehicle_width_m) / bbox_w
                if math.isfinite(z_from_width):
                    z_m = 0.8 * z_m + 0.2 * z_from_width
            except Exception:
                pass

        return np.array([z_m, y_lateral], dtype=float)

    def _create_track(
        self,
        track_id: int,
        t_meas: float,
        z_xy: np.ndarray,
        class_name: str,
        box_xyxy: Optional[List[float]],
    ) -> None:
        x0 = np.array([z_xy[0], z_xy[1], 0.0, 0.0, 0.0], dtype=float)
        P0 = np.diag([4.0, 4.0, np.deg2rad(35.0) ** 2, 9.0, np.deg2rad(20.0) ** 2])

        ekf = UnicycleEKF(
            EKFConfig(
                dt=self.dt_default,
                gate_threshold=self.gate_threshold,
                sigma_proc_x=self.sigma_proc_x,
                sigma_proc_y=self.sigma_proc_y,
                sigma_proc_psi=math.radians(self.sigma_proc_psi_deg),
                sigma_proc_v=self.sigma_proc_v,
                sigma_proc_omega=math.radians(self.sigma_proc_omega_deg),
                sigma_meas_x=self.sigma_meas_x,
                sigma_meas_y=self.sigma_meas_y,
            ),
            x0,
            P0,
        )

        self.tracks[track_id] = TrackMemory(
            ekf=ekf,
            last_t=t_meas,
            last_meas_xy=z_xy.copy(),
            last_meas_heading=None,
            last_meas_speed=0.0,
            last_meas_omega=0.0,
            class_name=class_name,
            last_box_xyxy=box_xyxy,
        )

    def _estimate_control_and_turn_rate(
        self,
        prev_xy: np.ndarray,
        curr_xy: np.ndarray,
        prev_heading: Optional[float],
        prev_speed: float,
        dt: float,
    ) -> Tuple[np.ndarray, float, float, float]:
        dx = float(curr_xy[0] - prev_xy[0])
        dy = float(curr_xy[1] - prev_xy[1])

        dist = math.hypot(dx, dy)
        v_meas = dist / max(dt, 1e-6)

        if dist > 1e-6:
            psi_meas = math.atan2(dy, dx)
        else:
            psi_meas = prev_heading if prev_heading is not None else 0.0

        a = (v_meas - prev_speed) / max(dt, 1e-6)

        if prev_heading is None or v_meas < 1e-3:
            omega_meas = 0.0
        else:
            omega_meas = wrap_angle(psi_meas - prev_heading) / max(dt, 1e-6)

        # controllo usato dal modello: solo accelerazione.
        # omega viene mantenuto come stato e aggiornato con la misura.
        u = np.array([a], dtype=float)

        return u, psi_meas, v_meas, omega_meas

    def _inject_heading_and_omega(
        self,
        ekf: UnicycleEKF,
        psi_meas: float,
        omega_meas: float,
    ) -> None:
        # correzione soft su heading e yaw-rate, mantenendo lo schema semplice
        ekf.x[2] = wrap_angle(0.7 * ekf.x[2] + 0.3 * psi_meas)
        ekf.x[4] = 0.7 * ekf.x[4] + 0.3 * omega_meas

    def detections_callback(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"JSON non valido su {self.input_topic}: {e}")
            return

        header = data.get("header", {})
        t_meas = float(header.get("stamp", time.time()))
        detections = data.get("detections_xyz", [])

        filtered_tracks: List[Dict[str, Any]] = []

        for det in detections:
            track_id = det.get("track_id", None)
            class_name = str(det.get("class_name", ""))
            box_xyxy = det.get("box_xyxy", None)

            if track_id is None:
                continue
            try:
                track_id = int(track_id)
            except Exception:
                continue

            if not self._class_allowed(class_name):
                continue

            z_xy = self._measurement_from_detection(det)
            if z_xy is None:
                continue

            if track_id not in self.tracks:
                self._create_track(
                    track_id=track_id,
                    t_meas=t_meas,
                    z_xy=z_xy,
                    class_name=class_name,
                    box_xyxy=box_xyxy,
                )

            mem = self.tracks[track_id]
            dt = t_meas - mem.last_t
            dt = float(np.clip(dt, self.min_dt_sec, self.max_dt_sec))
            mem.ekf.set_dt(dt)

            u, psi_meas, v_meas, omega_meas = self._estimate_control_and_turn_rate(
                prev_xy=mem.last_meas_xy,
                curr_xy=z_xy,
                prev_heading=mem.last_meas_heading,
                prev_speed=mem.last_meas_speed,
                dt=dt,
            )

            mem.ekf.predict(u)
            self._inject_heading_and_omega(mem.ekf, psi_meas, omega_meas)
            step_result = mem.ekf.update(z_xy)
            state = np.asarray(step_result["state"], dtype=float)

            mem.last_t = t_meas
            mem.last_meas_xy = z_xy.copy()
            mem.last_meas_heading = psi_meas
            mem.last_meas_speed = v_meas
            mem.last_meas_omega = omega_meas
            mem.class_name = class_name
            mem.last_box_xyxy = box_xyxy

            filtered_tracks.append({
                "track_id": track_id,
                "class_name": class_name,
                "measurement_xy_m": [float(z_xy[0]), float(z_xy[1])],
                "state": {
                    "x_m": float(state[0]),
                    "y_m": float(state[1]),
                    "psi_rad": float(state[2]),
                    "v_mps": float(state[3]),
                    "omega_radps": float(state[4]),
                },
                "control": {
                    "a_mps2": float(u[0]),
                },
                "accepted": bool(step_result["accepted"]),
                "nis": float(step_result["nis"]) if not np.isnan(step_result["nis"]) else None,
                "box_xyxy": box_xyxy,
            })

        out = {
            "header": {
                "stamp": t_meas,
                "frame_id": header.get("frame_id", ""),
            },
            "tracks": filtered_tracks,
        }

        out_msg = String()
        out_msg.data = json.dumps(out, ensure_ascii=False)
        self.pub.publish(out_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = UnicycleKFTrackerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

