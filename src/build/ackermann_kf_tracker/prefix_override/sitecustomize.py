import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/isarlab/ros2_humble/src/install/ackermann_kf_tracker'
