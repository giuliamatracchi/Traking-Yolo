#!/usr/bin/env python3

from __future__ import annotations

import math
import os
from typing import Optional

import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node


def quat_to_yaw(q) -> float:
    x = q.x
    y = q.y
    z = q.z
    w = q.w
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class AmbulanceMotionNode(Node):
    """
    Publishes desired ambulance poses on /ambulance_position_des.

    Behaviour:
    - On first run, captures current odometry as reference pose and saves it.
    - On subsequent runs, reloads that same reference pose.
    - Holds the reference pose for reset_hold_sec seconds.
    - Then publishes a sampled trajectory at rate_hz.

    Supported motion types:
    - straight_right
    - sine_right
    - accelerated_right   (initial burst, then quick slowdown)
    """

    def __init__(self) -> None:
        super().__init__("ambulance_motion_node")

        self.declare_parameter("odom_topic", "/environment/GPS/GPSSDTAMBULANCE/odom")
        self.declare_parameter("cmd_topic", "/ambulance_position_des")
        self.declare_parameter("frame_id", "map")

        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("motion_type", "straight_right")
        self.declare_parameter("duration_sec", 20.0)
        self.declare_parameter("reset_hold_sec", 1.0)

        self.declare_parameter("reference_pose_file", "")

        # straight / sine
        self.declare_parameter("speed_mps", 1.0)
        self.declare_parameter("amplitude_m", 1.0)
        self.declare_parameter("frequency_hz", 0.15)

        # accelerated_right:
        # velocity profile = tail_speed + burst_speed * exp(-t / burst_tau_sec)
        # so it starts fast and slows quickly
        self.declare_parameter("burst_speed_mps", 5.0)
        self.declare_parameter("tail_speed_mps", 0.10)
        self.declare_parameter("burst_tau_sec", 0.35)

        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)

        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.motion_type = str(self.get_parameter("motion_type").value)
        self.duration_sec = float(self.get_parameter("duration_sec").value)
        self.reset_hold_sec = float(self.get_parameter("reset_hold_sec").value)

        self.reference_pose_file = str(self.get_parameter("reference_pose_file").value)

        self.speed_mps = float(self.get_parameter("speed_mps").value)
        self.amplitude_m = float(self.get_parameter("amplitude_m").value)
        self.frequency_hz = float(self.get_parameter("frequency_hz").value)

        self.burst_speed_mps = float(self.get_parameter("burst_speed_mps").value)
        self.tail_speed_mps = float(self.get_parameter("tail_speed_mps").value)
        self.burst_tau_sec = float(self.get_parameter("burst_tau_sec").value)

        self.pub = self.create_publisher(PoseStamped, self.cmd_topic, 10)
        self.sub = self.create_subscription(Odometry, self.odom_topic, self.odom_cb, 20)
        self.timer = self.create_timer(1.0 / self.rate_hz, self.timer_cb)

        self.current_odom: Optional[Odometry] = None
        self.reference_pose: Optional[dict] = None

        self.started = False
        self.start_time_sec: Optional[float] = None
        self.motion_start_time_sec: Optional[float] = None
        self.finished = False

        self.get_logger().info(
            "AmbulanceMotionNode started\n"
            f"  odom_topic           : {self.odom_topic}\n"
            f"  cmd_topic            : {self.cmd_topic}\n"
            f"  frame_id             : {self.frame_id}\n"
            f"  rate_hz              : {self.rate_hz}\n"
            f"  motion_type          : {self.motion_type}\n"
            f"  duration_sec         : {self.duration_sec}\n"
            f"  reset_hold_sec       : {self.reset_hold_sec}\n"
            f"  speed_mps            : {self.speed_mps}\n"
            f"  amplitude_m          : {self.amplitude_m}\n"
            f"  frequency_hz         : {self.frequency_hz}\n"
            f"  burst_speed_mps      : {self.burst_speed_mps}\n"
            f"  tail_speed_mps       : {self.tail_speed_mps}\n"
            f"  burst_tau_sec        : {self.burst_tau_sec}\n"
            f"  reference_pose_file  : {self.reference_pose_file}"
        )

    def odom_cb(self, msg: Odometry) -> None:
        self.current_odom = msg

        if self.reference_pose is None:
            loaded = self._try_load_reference_pose()
            if loaded is not None:
                self.reference_pose = loaded
                self.get_logger().info("Loaded reference pose from file.")
            else:
                self.reference_pose = self._reference_from_odom(msg)
                self._try_save_reference_pose(self.reference_pose)
                self.get_logger().info("Captured current odom as reference pose and saved it.")

    def _reference_from_odom(self, msg: Odometry) -> dict:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        return {
            "x": float(p.x),
            "y": float(p.y),
            "z": float(p.z),
            "qx": float(q.x),
            "qy": float(q.y),
            "qz": float(q.z),
            "qw": float(q.w),
            "yaw": float(quat_to_yaw(q)),
        }

    def _try_load_reference_pose(self) -> Optional[dict]:
        if not self.reference_pose_file:
            return None
        if not os.path.exists(self.reference_pose_file):
            return None

        try:
            with open(self.reference_pose_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            required = ["x", "y", "z", "qx", "qy", "qz", "qw", "yaw"]
            if all(k in data for k in required):
                return data
        except Exception as e:
            self.get_logger().warning(f"Could not load reference pose file: {e}")

        return None

    def _try_save_reference_pose(self, pose: dict) -> None:
        if not self.reference_pose_file:
            return

        try:
            os.makedirs(os.path.dirname(self.reference_pose_file), exist_ok=True)
            with open(self.reference_pose_file, "w", encoding="utf-8") as f:
                yaml.safe_dump(pose, f, sort_keys=False)
        except Exception as e:
            self.get_logger().warning(f"Could not save reference pose file: {e}")

    def _make_pose_msg(
        self, x: float, y: float, z: float, qx: float, qy: float, qz: float, qw: float
    ) -> PoseStamped:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        msg.pose.orientation.x = float(qx)
        msg.pose.orientation.y = float(qy)
        msg.pose.orientation.z = float(qz)
        msg.pose.orientation.w = float(qw)
        return msg

    def _publish_reference_pose(self) -> None:
        rp = self.reference_pose
        assert rp is not None
        msg = self._make_pose_msg(
            rp["x"], rp["y"], rp["z"],
            rp["qx"], rp["qy"], rp["qz"], rp["qw"]
        )
        self.pub.publish(msg)

    def _displacement_straight(self, t: float) -> float:
        return self.speed_mps * t

    def _displacement_accelerated_right(self, t: float) -> float:
        """
        Initial burst with quick slowdown:
        v(t) = tail_speed + burst_speed * exp(-t / tau)

        Integrating:
        s(t) = tail_speed * t + burst_speed * tau * (1 - exp(-t / tau))

        This gives a big initial motion, then quickly transitions
        to a much slower motion so the camera can perceive the jump.
        """
        tau = max(self.burst_tau_sec, 1e-3)
        return (
            self.tail_speed_mps * t
            + self.burst_speed_mps * tau * (1.0 - math.exp(-t / tau))
        )

    def timer_cb(self) -> None:
        if self.finished:
            return

        if self.current_odom is None or self.reference_pose is None:
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9

        if not self.started:
            self.started = True
            self.start_time_sec = now_sec
            self.motion_start_time_sec = now_sec + self.reset_hold_sec
            self.get_logger().info(
                f"Reset/hold phase for {self.reset_hold_sec:.2f} s, then motion starts."
            )

        assert self.motion_start_time_sec is not None

        # Reset phase: continuously republish initial pose
        if now_sec < self.motion_start_time_sec:
            self._publish_reference_pose()
            return

        t = now_sec - self.motion_start_time_sec
        if t >= self.duration_sec:
            t = self.duration_sec
            self.finished = True
            self.get_logger().info("Trajectory completed.")

        rp = self.reference_pose
        assert rp is not None

        x0 = rp["x"]
        y0 = rp["y"]
        z0 = rp["z"]
        qx = rp["qx"]
        qy = rp["qy"]
        qz = rp["qz"]
        qw = rp["qw"]
        yaw0 = rp["yaw"]

        # Local axes from initial yaw
        forward_x = math.cos(yaw0)
        forward_y = math.sin(yaw0)

        # Right direction in local frame
        right_x = math.sin(yaw0)
        right_y = -math.cos(yaw0)

        if self.motion_type == "straight_right":
            s = self._displacement_straight(t)
            x = x0 + s * right_x
            y = y0 + s * right_y

        elif self.motion_type == "sine_right":
            forward_s = self._displacement_straight(t)
            lateral = self.amplitude_m * math.sin(2.0 * math.pi * self.frequency_hz * t)
            x = x0 + forward_s * forward_x + lateral * right_x
            y = y0 + forward_s * forward_y + lateral * right_y

        elif self.motion_type == "accelerated_right":
            s = self._displacement_accelerated_right(t)
            x = x0 + s * right_x
            y = y0 + s * right_y

        else:
            self.get_logger().warning(
                f"Unknown motion_type='{self.motion_type}', fallback to straight_right."
            )
            s = self._displacement_straight(t)
            x = x0 + s * right_x
            y = y0 + s * right_y

        msg = self._make_pose_msg(x, y, z0, qx, qy, qz, qw)
        self.pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AmbulanceMotionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

