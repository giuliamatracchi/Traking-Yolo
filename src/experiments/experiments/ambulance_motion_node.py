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

    New simplified behavior:
    - On first run, captures current odometry as reference pose and saves it.
    - On subsequent runs, reloads that same reference pose.
    - Holds the reference pose for reset_hold_sec seconds.
    - Then moves in a straight line to the RIGHT of the initial heading.
    - Stops after travel_distance_m meters.
    - No cyclic respawn/teleport during motion.
    """

    def __init__(self) -> None:
        super().__init__("ambulance_motion_node")

        self.declare_parameter("odom_topic", "/environment/GPS/GPSSDTAMBULANCE/odom")
        self.declare_parameter("cmd_topic", "/ambulance_position_des")
        self.declare_parameter("frame_id", "map")

        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("reset_hold_sec", 1.0)
        self.declare_parameter("reference_pose_file", "")

        # Straight motion only
        self.declare_parameter("speed_mps", 0.3)
        self.declare_parameter("travel_distance_m", 20.0)

        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)

        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.reset_hold_sec = float(self.get_parameter("reset_hold_sec").value)
        self.reference_pose_file = str(self.get_parameter("reference_pose_file").value)

        self.speed_mps = float(self.get_parameter("speed_mps").value)
        self.travel_distance_m = float(self.get_parameter("travel_distance_m").value)

        self.pub = self.create_publisher(PoseStamped, self.cmd_topic, 10)
        self.sub = self.create_subscription(Odometry, self.odom_topic, self.odom_cb, 20)
        self.timer = self.create_timer(1.0 / self.rate_hz, self.timer_cb)

        self.current_odom: Optional[Odometry] = None
        self.reference_pose: Optional[dict] = None

        self.started = False
        self.start_time_sec: Optional[float] = None
        self.motion_start_time_sec: Optional[float] = None
        self.finished = False
        self.final_pose_published = False

        self.get_logger().info(
            "AmbulanceMotionNode started\n"
            f"  odom_topic           : {self.odom_topic}\n"
            f"  cmd_topic            : {self.cmd_topic}\n"
            f"  frame_id             : {self.frame_id}\n"
            f"  rate_hz              : {self.rate_hz}\n"
            f"  reset_hold_sec       : {self.reset_hold_sec}\n"
            f"  speed_mps            : {self.speed_mps}\n"
            f"  travel_distance_m    : {self.travel_distance_m}\n"
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
            directory = os.path.dirname(self.reference_pose_file)
            if directory:
                os.makedirs(directory, exist_ok=True)
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

    def _publish_final_pose(self) -> None:
        rp = self.reference_pose
        assert rp is not None

        yaw0 = rp["yaw"]
        right_x = math.sin(yaw0)
        right_y = -math.cos(yaw0)

        x = rp["x"] + self.travel_distance_m * right_x
        y = rp["y"] + self.travel_distance_m * right_y

        msg = self._make_pose_msg(
            x, y, rp["z"],
            rp["qx"], rp["qy"], rp["qz"], rp["qw"]
        )
        self.pub.publish(msg)

    def timer_cb(self) -> None:
        if self.current_odom is None or self.reference_pose is None:
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9

        if not self.started:
            self.started = True
            self.start_time_sec = now_sec
            self.motion_start_time_sec = now_sec + self.reset_hold_sec
            self.get_logger().info(
                f"Reset/hold phase for {self.reset_hold_sec:.2f} s, then straight-right motion starts."
            )

        assert self.motion_start_time_sec is not None

        if now_sec < self.motion_start_time_sec:
            self._publish_reference_pose()
            return

        if self.finished:
            if not self.final_pose_published:
                self._publish_final_pose()
                self.final_pose_published = True
                self.get_logger().info("Final pose published. Motion completed.")
            return

        t = now_sec - self.motion_start_time_sec
        s = self.speed_mps * t

        if s >= self.travel_distance_m:
            s = self.travel_distance_m
            self.finished = True

        rp = self.reference_pose
        assert rp is not None

        yaw0 = rp["yaw"]
        right_x = math.sin(yaw0)
        right_y = -math.cos(yaw0)

        x = rp["x"] + s * right_x
        y = rp["y"] + s * right_y

        msg = self._make_pose_msg(
            x, y, rp["z"],
            rp["qx"], rp["qy"], rp["qz"], rp["qw"]
        )
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

