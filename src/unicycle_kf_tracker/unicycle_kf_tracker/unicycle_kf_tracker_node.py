import rclpy
from rclpy.node import Node


class UnicycleKFTrackerNode(Node):
    def __init__(self):
        super().__init__('unicycle_kf_tracker')
        self.get_logger().info('Unicycle KF tracker avviato correttamente')


def main(args=None):
    rclpy.init(args=args)
    node = UnicycleKFTrackerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

