#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import math
import os
from bisect import bisect_left
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def rotation_2d(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


@dataclass
class TimedPose2D:
    t: float
    x: float
    y: float
    yaw: float = 0.0


@dataclass
class MetricSample:
    t: float
    stage: str
    target_id: int
    gt_x: float
    gt_y: float
    est_x: float
    est_y: float
    err_x: float
    err_y: float
    err_norm: float


class TrajectoryMetricsNode(Node):
    def __init__(self) -> None:
        super().__init__("trajectory_metrics_node")

        self.declare_parameter("stage", "ackermann")

        self.declare_parameter("camera_gps_topic", "/environment/GPS/GPSSDT/odom")
        self.declare_parameter("ambulance_gps_topic", "/environment/GPS/GPSSDTAMBULANCE/odom")

        self.declare_parameter("fusion_topic", "/yolo/detections_xyz")
        self.declare_parameter("ackermann_topic", "/ackermann_kf/tracks")
        self.declare_parameter("unicycle_topic", "/unicycle_kf/tracks")

        self.declare_parameter("target_track_id", -1)
        self.declare_parameter("allowed_classes", ["truck", "ambulance", "*"])

        self.declare_parameter("image_width_px", 1280)
        self.declare_parameter("horizontal_fov_deg", 90.0)

        self.declare_parameter("max_time_offset_sec", 0.30)
        self.declare_parameter("output_dir", "metrics_output")
        self.declare_parameter("save_csv", True)
        self.declare_parameter("save_plots", True)

        self.stage = str(self.get_parameter("stage").value).strip().lower()

        self.camera_gps_topic = str(self.get_parameter("camera_gps_topic").value)
        self.ambulance_gps_topic = str(self.get_parameter("ambulance_gps_topic").value)

        self.fusion_topic = str(self.get_parameter("fusion_topic").value)
        self.ackermann_topic = str(self.get_parameter("ackermann_topic").value)
        self.unicycle_topic = str(self.get_parameter("unicycle_topic").value)

        self.target_track_id = int(self.get_parameter("target_track_id").value)
        allowed = self.get_parameter("allowed_classes").value
        self.allowed_classes = set(str(x) for x in allowed) if allowed else {"*"}

        self.image_width_px = int(self.get_parameter("image_width_px").value)
        self.horizontal_fov_deg = float(self.get_parameter("horizontal_fov_deg").value)

        self.max_time_offset_sec = float(self.get_parameter("max_time_offset_sec").value)
        self.output_dir = str(self.get_parameter("output_dir").value)
        self.save_csv = bool(self.get_parameter("save_csv").value)
        self.save_plots = bool(self.get_parameter("save_plots").value)

        if self.stage not in {"fusion", "ackermann", "unicycle"}:
            raise RuntimeError("stage deve essere uno tra: fusion, ackermann, unicycle")

        hfov_rad = math.radians(self.horizontal_fov_deg)
        self.focal_px = self.image_width_px / (2.0 * math.tan(hfov_rad / 2.0))

        os.makedirs(self.output_dir, exist_ok=True)

        self.camera_history: List[TimedPose2D] = []
        self.ambulance_history: List[TimedPose2D] = []
        self.camera_times: List[float] = []
        self.ambulance_times: List[float] = []

        self.first_gt_rel_cam: Optional[np.ndarray] = None

        self.selected_target_id: Optional[int] = None
        self.first_est_raw: Optional[np.ndarray] = None
        self.rotation_gt_to_est: Optional[np.ndarray] = None

        self.metric_samples: List[MetricSample] = []

        self.gt_initialized_logged = False
        self.est_initialized_logged = False

        self.sub_cam = self.create_subscription(
            Odometry, self.camera_gps_topic, self.camera_odom_callback, 20
        )
        self.sub_amb = self.create_subscription(
            Odometry, self.ambulance_gps_topic, self.ambulance_odom_callback, 20
        )
        self.sub_est = self.create_subscription(
            String, self._estimation_topic(), self.estimation_callback, 20
        )

        self.get_logger().info(
            "TrajectoryMetricsNode ready\n"
            f"  stage: {self.stage}\n"
            f"  camera_gps_topic: {self.camera_gps_topic}\n"
            f"  ambulance_gps_topic: {self.ambulance_gps_topic}\n"
            f"  estimation_topic: {self._estimation_topic()}\n"
            f"  target_track_id: {self.target_track_id}\n"
            f"  output_dir: {self.output_dir}"
        )

    def _estimation_topic(self) -> str:
        if self.stage == "fusion":
            return self.fusion_topic
        if self.stage == "ackermann":
            return self.ackermann_topic
        return self.unicycle_topic

    def _append_pose(self, history: List[TimedPose2D], times: List[float], pose: TimedPose2D) -> None:
        history.append(pose)
        times.append(pose.t)

        max_len = 5000
        if len(history) > max_len:
            del history[: len(history) - max_len]
            del times[: len(times) - max_len]

    def camera_odom_callback(self, msg: Odometry) -> None:
        t = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
        self._append_pose(
            self.camera_history,
            self.camera_times,
            TimedPose2D(t=t, x=float(p.x), y=float(p.y), yaw=yaw),
        )

    def ambulance_odom_callback(self, msg: Odometry) -> None:
        t = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
        self._append_pose(
            self.ambulance_history,
            self.ambulance_times,
            TimedPose2D(t=t, x=float(p.x), y=float(p.y), yaw=yaw),
        )

    def _find_closest_pose(
        self,
        history: List[TimedPose2D],
        times: List[float],
        t_query: float,
    ) -> Optional[TimedPose2D]:
        if not times:
            return None

        idx = bisect_left(times, t_query)
        candidates = []

        if idx < len(history):
            candidates.append(history[idx])
        if idx > 0:
            candidates.append(history[idx - 1])

        if not candidates:
            return None

        best = min(candidates, key=lambda p: abs(p.t - t_query))
        if abs(best.t - t_query) > self.max_time_offset_sec:
            return None

        return best

    def _compute_gt_relative_in_camera_frame(self, t_est: float) -> Optional[np.ndarray]:
        cam = self._find_closest_pose(self.camera_history, self.camera_times, t_est)
        amb = self._find_closest_pose(self.ambulance_history, self.ambulance_times, t_est)

        if cam is None or amb is None:
            return None

        d_world = np.array([amb.x - cam.x, amb.y - cam.y], dtype=float)

        r_wc = rotation_2d(-cam.yaw)
        d_cam = r_wc @ d_world

        if self.first_gt_rel_cam is None:
            self.first_gt_rel_cam = d_cam.copy()
            if not self.gt_initialized_logged:
                self.get_logger().info(
                    f"GT inizializzata: first_gt_rel_cam = "
                    f"[{self.first_gt_rel_cam[0]:.3f}, {self.first_gt_rel_cam[1]:.3f}]"
                )
                self.gt_initialized_logged = True

        return d_cam - self.first_gt_rel_cam

    def _class_allowed(self, class_name: Optional[str]) -> bool:
        if "*" in self.allowed_classes:
            return True
        if class_name is None:
            return False
        return str(class_name) in self.allowed_classes

    def _extract_estimate_xy(self, item: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
        if self.stage == "fusion":
            z_raw = item.get("z_m")
            center_xy = item.get("center_xy")

            if z_raw is None or center_xy is None or len(center_xy) != 2:
                return None, None

            try:
                x_forward = float(z_raw) / 100.0
                cx = float(center_xy[0])
            except Exception:
                return None, None

            y_lateral = ((cx - (self.image_width_px / 2.0)) * x_forward) / self.focal_px
            return x_forward, y_lateral

        state = item.get("state", {})
        if not isinstance(state, dict):
            return None, None

        x_m = state.get("x_m")
        y_m = state.get("y_m")
        if x_m is None or y_m is None:
            return None, None

        try:
            return float(x_m), float(y_m)
        except Exception:
            return None, None

    def _choose_target(self, items: List[Dict[str, Any]]) -> Optional[Tuple[int, Dict[str, Any]]]:
        if not items:
            return None

        if self.selected_target_id is not None:
            for item in items:
                tid = item.get("track_id")
                if tid is not None and int(tid) == self.selected_target_id:
                    return self.selected_target_id, item
            return None

        if self.target_track_id >= 0:
            for item in items:
                tid = item.get("track_id")
                if tid is not None and int(tid) == self.target_track_id:
                    self.selected_target_id = self.target_track_id
                    return self.selected_target_id, item
            return None

        candidates = []
        for item in items:
            tid = item.get("track_id")
            if tid is None:
                continue

            class_name = item.get("class_name")
            if not self._class_allowed(class_name):
                continue

            est_xy = self._extract_estimate_xy(item)
            if est_xy[0] is None or est_xy[1] is None:
                continue

            x_est = float(est_xy[0])
            candidates.append((int(tid), item, x_est))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[2])
        self.selected_target_id = candidates[0][0]
        self.get_logger().info(f"Target selezionato automaticamente: track_id={self.selected_target_id}")
        return self.selected_target_id, candidates[0][1]

    def _extract_stamp_from_json(self, data: Dict[str, Any]) -> float:
        header = data.get("header", {})
        stamp = header.get("stamp")

        if isinstance(stamp, dict):
            sec = float(stamp.get("sec", 0.0))
            nanosec = float(stamp.get("nanosec", 0.0))
            return sec + nanosec * 1e-9

        if isinstance(stamp, (int, float)):
            return float(stamp)

        return self.get_clock().now().nanoseconds * 1e-9

    def _initialize_est_alignment(self, est_rel: np.ndarray, gt_rel: np.ndarray) -> None:
        if self.first_est_raw is None:
            self.first_est_raw = est_rel.copy()
            if not self.est_initialized_logged:
                self.get_logger().info(
                    f"Stima inizializzata: first_est_raw = "
                    f"[{self.first_est_raw[0]:.3f}, {self.first_est_raw[1]:.3f}]"
                )
                self.est_initialized_logged = True

        est_centered = est_rel - self.first_est_raw

        if self.rotation_gt_to_est is None:
            norm_gt = np.linalg.norm(gt_rel)
            norm_est = np.linalg.norm(est_centered)

            if norm_gt > 1e-6 and norm_est > 1e-6:
                ang_gt = math.atan2(gt_rel[1], gt_rel[0])
                ang_est = math.atan2(est_centered[1], est_centered[0])
                theta = ang_est - ang_gt
                self.rotation_gt_to_est = rotation_2d(theta)

                self.get_logger().info(
                    f"Allineamento GT->stima inizializzato: theta = {math.degrees(theta):.2f} deg"
                )

    def estimation_callback(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"JSON non valido sul topic stima: {e}")
            return

        t_est = self._extract_stamp_from_json(data)
        gt_rel = self._compute_gt_relative_in_camera_frame(t_est)
        if gt_rel is None:
            return

        items = data.get("detections_xyz", []) if self.stage == "fusion" else data.get("tracks", [])
        chosen = self._choose_target(items)
        if chosen is None:
            return

        target_id, item = chosen
        est_xy = self._extract_estimate_xy(item)
        if est_xy[0] is None or est_xy[1] is None:
            return

        est_raw = np.array([float(est_xy[0]), float(est_xy[1])], dtype=float)

        if self.first_est_raw is None:
            self.first_est_raw = est_raw.copy()
            self.get_logger().info(
                f"Primo campione stima per target {target_id}: "
                f"[{self.first_est_raw[0]:.3f}, {self.first_est_raw[1]:.3f}]"
            )

        est_centered = est_raw - self.first_est_raw

        if self.rotation_gt_to_est is None:
            norm_gt = np.linalg.norm(gt_rel)
            norm_est = np.linalg.norm(est_centered)
            if norm_gt > 1e-6 and norm_est > 1e-6:
                ang_gt = math.atan2(gt_rel[1], gt_rel[0])
                ang_est = math.atan2(est_centered[1], est_centered[0])
                theta = ang_est - ang_gt
                self.rotation_gt_to_est = rotation_2d(theta)
                self.get_logger().info(
                    f"Rotazione iniziale GT->stima fissata: {math.degrees(theta):.2f} deg"
                )

        if self.rotation_gt_to_est is not None:
            gt_aligned = self.rotation_gt_to_est @ gt_rel
        else:
            gt_aligned = gt_rel.copy()

        err = est_centered - gt_aligned
        err_norm = float(np.linalg.norm(err))

        self.metric_samples.append(
            MetricSample(
                t=t_est,
                stage=self.stage,
                target_id=target_id,
                gt_x=float(gt_aligned[0]),
                gt_y=float(gt_aligned[1]),
                est_x=float(est_centered[0]),
                est_y=float(est_centered[1]),
                err_x=float(err[0]),
                err_y=float(err[1]),
                err_norm=err_norm,
            )
        )

    def _summary_dict(self) -> Dict[str, float]:
        if not self.metric_samples:
            return {
                "n_samples": 0,
                "mae_x": float("nan"),
                "mae_y": float("nan"),
                "mae_norm": float("nan"),
                "rmse_x": float("nan"),
                "rmse_y": float("nan"),
                "rmse_norm": float("nan"),
                "final_error": float("nan"),
                "max_error": float("nan"),
            }

        err_x = np.array([s.err_x for s in self.metric_samples], dtype=float)
        err_y = np.array([s.err_y for s in self.metric_samples], dtype=float)
        err_n = np.array([s.err_norm for s in self.metric_samples], dtype=float)

        return {
            "n_samples": int(len(self.metric_samples)),
            "mae_x": float(np.mean(np.abs(err_x))),
            "mae_y": float(np.mean(np.abs(err_y))),
            "mae_norm": float(np.mean(np.abs(err_n))),
            "rmse_x": float(np.sqrt(np.mean(err_x ** 2))),
            "rmse_y": float(np.sqrt(np.mean(err_y ** 2))),
            "rmse_norm": float(np.sqrt(np.mean(err_n ** 2))),
            "final_error": float(err_n[-1]),
            "max_error": float(np.max(err_n)),
        }

    def save_outputs(self) -> None:
        if not self.metric_samples:
            self.get_logger().warn("Nessun campione raccolto, niente da salvare.")
            return

        summary = self._summary_dict()

        detail_csv = os.path.join(self.output_dir, f"{self.stage}_metrics_detail.csv")
        summary_csv = os.path.join(self.output_dir, f"{self.stage}_metrics_summary.csv")
        traj_png = os.path.join(self.output_dir, f"{self.stage}_trajectory.png")
        err_png = os.path.join(self.output_dir, f"{self.stage}_error.png")

        if self.save_csv:
            with open(detail_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "t", "stage", "target_id",
                    "gt_x", "gt_y",
                    "est_x", "est_y",
                    "err_x", "err_y", "err_norm",
                ])
                for s in self.metric_samples:
                    writer.writerow([
                        s.t, s.stage, s.target_id,
                        s.gt_x, s.gt_y,
                        s.est_x, s.est_y,
                        s.err_x, s.err_y, s.err_norm,
                    ])

            with open(summary_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "stage", "target_id", "n_samples",
                    "mae_x", "mae_y", "mae_norm",
                    "rmse_x", "rmse_y", "rmse_norm",
                    "final_error", "max_error",
                ])
                writer.writerow([
                    self.stage,
                    self.selected_target_id if self.selected_target_id is not None else -1,
                    summary["n_samples"],
                    summary["mae_x"],
                    summary["mae_y"],
                    summary["mae_norm"],
                    summary["rmse_x"],
                    summary["rmse_y"],
                    summary["rmse_norm"],
                    summary["final_error"],
                    summary["max_error"],
                ])

        if self.save_plots:
            t = np.array([s.t for s in self.metric_samples], dtype=float)
            t = t - t[0]

            gt_x = np.array([s.gt_x for s in self.metric_samples], dtype=float)
            gt_y = np.array([s.gt_y for s in self.metric_samples], dtype=float)
            est_x = np.array([s.est_x for s in self.metric_samples], dtype=float)
            est_y = np.array([s.est_y for s in self.metric_samples], dtype=float)
            err_n = np.array([s.err_norm for s in self.metric_samples], dtype=float)

            plt.figure(figsize=(8, 6))
            plt.plot(gt_x, gt_y, label="GT relativa (GPS)")
            plt.plot(est_x, est_y, label=f"Stima {self.stage}")
            plt.xlabel("x [m]")
            plt.ylabel("y [m]")
            plt.title(f"Traiettoria relativa - {self.stage}")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(traj_png, dpi=150)
            plt.close()

            plt.figure(figsize=(8, 4))
            plt.plot(t, err_n, label="Errore euclideo")
            plt.xlabel("t [s]")
            plt.ylabel("errore [m]")
            plt.title(f"Errore nel tempo - {self.stage}")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(err_png, dpi=150)
            plt.close()

        self.get_logger().info(
            f"Salvati:\n"
            f"  {detail_csv}\n"
            f"  {summary_csv}\n"
            f"  {traj_png}\n"
            f"  {err_png}"
        )

        self.get_logger().info(
            f"Summary {self.stage}: "
            f"target_id={self.selected_target_id}, "
            f"n={summary['n_samples']}, "
            f"MAE_norm={summary['mae_norm']:.3f} m, "
            f"RMSE_norm={summary['rmse_norm']:.3f} m, "
            f"final_error={summary['final_error']:.3f} m, "
            f"max_error={summary['max_error']:.3f} m"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrajectoryMetricsNode()

    try:
        rclpy.spin(node)
    finally:
        try:
            node.save_outputs()
        except Exception as e:
            node.get_logger().error(f"Errore nel salvataggio output: {e}")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

