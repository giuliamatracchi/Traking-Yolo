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


def numerical_jacobian(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y0 = np.asarray(f(x), dtype=float)
    n = x.size
    m = y0.size
    J = np.zeros((m, n), dtype=float)

    for i in range(n):
        dx = np.zeros(n, dtype=float)
        dx[i] = eps
        yp = np.asarray(f(x + dx), dtype=float)
        ym = np.asarray(f(x - dx), dtype=float)
        J[:, i] = (yp - ym) / (2.0 * eps)

    return J


@dataclass
class VehicleParams:
    lf: float = 0.75
    lr: float = 0.75
    max_steer_rad: float = np.deg2rad(35.0)

    @property
    def wheelbase(self) -> float:
        return self.lf + self.lr


@dataclass
class EKFConfig:
    dt: float = 0.1
    gate_threshold: float = 9.21
    sigma_proc_x: float = 0.10
    sigma_proc_y: float = 0.10
    sigma_proc_psi: float = np.deg2rad(2.0)
    sigma_proc_v: float = 0.75
    sigma_meas_x: float = 0.80
    sigma_meas_y: float = 0.80

    def Q(self) -> np.ndarray:
        return np.diag([
            self.sigma_proc_x ** 2,
            self.sigma_proc_y ** 2,
            self.sigma_proc_psi ** 2,
            self.sigma_proc_v ** 2,
        ])

    def R(self) -> np.ndarray:
        return np.diag([
            self.sigma_meas_x ** 2,
            self.sigma_meas_y ** 2,
        ])


class KinematicBicycleEKF:

    def __init__(self, params: VehicleParams, config: EKFConfig, x0: np.ndarray, P0: np.ndarray):

        self.params = params
        self.config = config

        self.x = np.asarray(x0, dtype=float).reshape(4)
        self.P = np.asarray(P0, dtype=float).reshape(4, 4)

        self.Q = config.Q()
        self.R = config.R()

        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ])

        self.I = np.eye(4)

    def set_dt(self, dt: float):
        self.config.dt = float(dt)

    def beta_from_delta(self, delta: float) -> float:

        delta = float(np.clip(delta, -self.params.max_steer_rad, self.params.max_steer_rad))

        ratio = self.params.lr / self.params.wheelbase

        return math.atan(ratio * math.tan(delta))

    def f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:

        X, Y, psi, v = x
        a, delta = u
        dt = self.config.dt

        beta = self.beta_from_delta(delta)

        Xn = X + dt * v * math.cos(psi + beta)
        Yn = Y + dt * v * math.sin(psi + beta)
        psin = wrap_angle(psi + dt * (v / self.params.lr) * math.sin(beta))
        vn = v + dt * a

        return np.array([Xn, Yn, psin, vn])

    def predict(self, u: np.ndarray):

        u = np.asarray(u).reshape(2)

        def fx(xx):
            return self.f(xx, u)

        F = numerical_jacobian(fx, self.x)

        self.x = self.f(self.x, u)

        self.P = F @ self.P @ F.T + self.Q

        self.x[2] = wrap_angle(self.x[2])

    def innovation(self, z: np.ndarray):

        z = np.asarray(z).reshape(2)

        z_pred = self.H @ self.x

        y = z - z_pred

        S = self.H @ self.P @ self.H.T + self.R

        return y, S

    def update(self, z: np.ndarray):

        y, S = self.innovation(z)

        Sinv = np.linalg.inv(S)

        nis = float(y.T @ Sinv @ y)

        if nis > self.config.gate_threshold:
            return {"accepted": False, "nis": nis, "state": self.x}

        K = self.P @ self.H.T @ Sinv

        self.x = self.x + K @ y

        self.x[2] = wrap_angle(self.x[2])

        I_KH = self.I - K @ self.H

        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        return {"accepted": True, "nis": nis, "state": self.x}

    def step(self, u: np.ndarray, z: Optional[np.ndarray]):

        self.predict(u)

        if z is None:
            return {"accepted": False, "nis": float("nan"), "state": self.x}

        return self.update(z)


@dataclass
class TrackMemory:

    ekf: KinematicBicycleEKF
    last_t: float
    last_meas_xy: np.ndarray
    last_meas_heading: Optional[float] = None
    last_meas_speed: float = 0.0
    class_name: str = ""
    last_box_xyxy: Optional[List[float]] = None


class AckermannKFTrackerNode(Node):

    def __init__(self):

        super().__init__("ackermann_kf_tracker")

        self.declare_parameter("input_topic", "yolo/detections_xyz")
        self.declare_parameter("output_topic", "ackermann_kf/tracks")
        self.declare_parameter("allowed_classes", ["*"])

        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value

        allowed = self.get_parameter("allowed_classes").value

        self.allowed_classes = set(str(x) for x in allowed) if allowed else set()

        self.tracks: Dict[int, TrackMemory] = {}

        self.sub = self.create_subscription(
            String,
            self.input_topic,
            self.detections_callback,
            10
        )

        self.pub = self.create_publisher(
            String,
            self.output_topic,
            10
        )

        self.get_logger().info(
            f"Ackermann tracker ready\n"
            f" input_topic: {self.input_topic}\n"
            f" output_topic: {self.output_topic}\n"
            f" allowed_classes: {self.allowed_classes}"
        )

    # 🔧 CORREZIONE WILDCARD
    def _class_allowed(self, class_name: Optional[str]) -> bool:

        if not self.allowed_classes:
            return True

        if "*" in self.allowed_classes:
            return True

        if class_name is None:
            return False

        return str(class_name) in self.allowed_classes

    def detections_callback(self, msg: String):

        data = json.loads(msg.data)

        header = data.get("header", {})

        detections = data.get("detections_xyz", [])

        tracks_out = []

        for det in detections:

            track_id = det.get("track_id")

            class_name = det.get("class_name")

            if not self._class_allowed(class_name):
                continue

            tracks_out.append(det)

        out = {
            "header": header,
            "tracks": tracks_out
        }

        out_msg = String()

        out_msg.data = json.dumps(out)

        self.pub.publish(out_msg)


def main(args=None):

    rclpy.init(args=args)

    node = AckermannKFTrackerNode()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()

