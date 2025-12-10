from .sensor import Sensor
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import struct
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class Lidar(Sensor):
    def __init__(self, node, env_topic, obs_settings, scan_frame: str = 'base_link', topic_scan: str = '/points', publish: bool = True, queue_size: int = 10, **kwargs):
        super().__init__()

        
        self._node = node
        self._env_topic = env_topic
        self._obs_settings = obs_settings or {}
        self._unreal_settings = kwargs.get('unreal_settings', {}) or {}
        self._ue = kwargs.get('environment_ue')
        self._ue_sensor = kwargs.get('ue_sensor')
        # default sensor type should reflect this class
        self._sensor_type = kwargs.get('sensor_type', 'Lidar')
        self._specific_name = kwargs.get('specific_name', 'default')

        # conversione: Unreal units -> metri (default UE uses cm -> 100 per metro)
        self._METERS_TO_UNREAL_UNIT = float(self._unreal_settings.get('METERS_TO_UNREAL_UNIT', self._unreal_settings.get('__METERS_TO_UNREAL_UNIT', 100.0)))

        # Lidar UE5 è 3D per default; rispetta solo un esplicito is_2d in obs_settings
        self._is_2d = bool(self._obs_settings.get('is_2d') or self._obs_settings.get('is2d') or False)

        # Frame, topic, publish flag, queue
        self.scan_frame = kwargs.get('scan_frame', scan_frame) or scan_frame
        self._topic_scan = kwargs.get('scan_topic', topic_scan) or topic_scan
        self._publish = bool(kwargs.get('publish', publish))
        self._queue_size = int(kwargs.get('queue_size', queue_size))

        self._expects_data = True

        # Helper per comporre topic
        def _join_topic(*parts: str) -> str:
            cleaned = [str(p).strip().strip('/') for p in parts if p is not None and str(p).strip() != ""]
            return "/".join(cleaned)

        base_topic = _join_topic(self._env_topic, self._sensor_type, self._specific_name)
        scan_topic_full = _join_topic(base_topic, self._topic_scan)
        # store point topic for future reconfiguration
        self._point_topic = scan_topic_full

        # Publisher PointCloud2
        self._pub_scan = None
        if self._publish and self._node is not None:
            try:
                qos = None
                try:
                    from rclpy.qos import qos_profile_sensor_data, QoSProfile
                    qos = qos_profile_sensor_data
                except Exception:
                    from rclpy.qos import QoSProfile
                    qos = QoSProfile(depth=self._queue_size)
                self._pub_scan = self._node.create_publisher(PointCloud2, self._point_topic, qos)
            except Exception:
                try:
                    self._node.get_logger().warning(f"Lidar: create PointCloud2 publisher failed for {self._point_topic}")
                except Exception:
                    pass
                self._pub_scan = None

        # (Removed leftover LaserScan scaffolding — Lidar publishes PointCloud2)

        # TF broadcaster (opzionale)
        self._tf_broadcaster = None
        try:
            self._tf_broadcaster = TransformBroadcaster(self._node)
        except Exception:
            self._tf_broadcaster = None

    def change_settings(self, **kwargs):
       pass

    def publish_observation(self, data=None):
        if data is None:
            return

        pts = []

        # normalize dict-like containers
        if isinstance(data, dict):
            for k in ('points', 'data'):
                if k in data:
                    data = data[k]
                    break

        # numpy-like
        if hasattr(data, 'shape') and hasattr(data, 'dtype'):
            arr = np.asarray(data)
            # if HxWx3 image-shaped, flatten
            if arr.ndim >= 2 and arr.shape[-1] >= 3:
                # collapse leading dims
                reshaped = arr.reshape(-1, arr.shape[-1])
                # take first three columns
                xyz = reshaped[:, :3].astype(np.float32)
                # scale from unreal units to metres
                xyz = xyz / float(self._METERS_TO_UNREAL_UNIT)
                # intensity as euclidean norm
                intensity = np.linalg.norm(xyz[:, :3], axis=1).astype(np.float32)
                # mask finite
                mask = np.isfinite(xyz).all(axis=1)
                xyz = xyz[mask]
                intensity = intensity[mask]
                if xyz.size == 0:
                    return
                pts_arr = np.empty((xyz.shape[0], 4), dtype=np.float32)
                pts_arr[:, 0:3] = xyz
                pts_arr[:, 3] = intensity
            elif arr.ndim == 1:
                # sequence of tuples/points
                try:
                    lst = [tuple(el) for el in arr]
                except Exception:
                    lst = []
                pts_arr = []
                for el in lst:
                    if len(el) >= 3:
                        x, y, z = float(el[0]), float(el[1]), float(el[2])
                        if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                            pts_arr.append((x / float(self._METERS_TO_UNREAL_UNIT), y / float(self._METERS_TO_UNREAL_UNIT), z / float(self._METERS_TO_UNREAL_UNIT)))
                if not pts_arr:
                    return
                pts_arr = np.array(pts_arr, dtype=np.float32)
                intensity = np.linalg.norm(pts_arr[:, :3], axis=1).astype(np.float32)
                pts_arr = np.column_stack((pts_arr, intensity))
            else:
                return

        elif isinstance(data, (list, tuple)):
            # list of points
            pts_list = []
            for el in data:
                try:
                    if isinstance(el, (list, tuple)) and len(el) >= 3:
                        x, y, z = float(el[0]), float(el[1]), float(el[2])
                        if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                                pts_list.append((x / float(self._METERS_TO_UNREAL_UNIT), y / float(self._METERS_TO_UNREAL_UNIT), z / float(self._METERS_TO_UNREAL_UNIT)))
                except Exception:
                    continue
            if not pts_list:
                return
            pts_arr = np.array(pts_list, dtype=np.float32)
            intensity = np.linalg.norm(pts_arr[:, :3], axis=1).astype(np.float32)
            pts_arr = np.column_stack((pts_arr, intensity))

        else:
            return

        # ensure we have Nx4 float32 array
        if isinstance(pts_arr, list):
            pts_arr = np.array(pts_arr, dtype=np.float32)
        if pts_arr.ndim != 2 or pts_arr.shape[1] < 3:
            return
        if pts_arr.shape[1] == 3:
            intensity = np.linalg.norm(pts_arr[:, :3], axis=1).astype(np.float32)
            pts_arr = np.column_stack((pts_arr, intensity))

        pts_arr = self._normalize_points(data)
        if pts_arr is None:
            return
        pc = self._build_pointcloud2(pts_arr)
        if self._pub_scan is not None:
            try:
                self._pub_scan.publish(pc)
            except Exception:
                try:
                    self._node.get_logger().warning('Lidar: failed to publish PointCloud2')
                except Exception:
                    pass
         
    def _normalize_points(self, data):
        if data is None:
            return None

        # dict-like: extract payload
        if isinstance(data, dict):
            for k in ('points', 'data'):
                if k in data:
                    data = data[k]
                    break

        pts_arr = None

        # numpy-like
        if hasattr(data, 'shape') and hasattr(data, 'dtype'):
            arr = np.asarray(data)
            if arr.ndim >= 2 and arr.shape[-1] >= 3:
                reshaped = arr.reshape(-1, arr.shape[-1])
                xyz = reshaped[:, :3].astype(np.float32)
                xyz = xyz / float(self._METERS_TO_UNREAL_UNIT)
                intensity = np.linalg.norm(xyz[:, :3], axis=1).astype(np.float32)
                mask = np.isfinite(xyz).all(axis=1)
                xyz = xyz[mask]
                intensity = intensity[mask]
                if xyz.size == 0:
                    return None
                pts_arr = np.empty((xyz.shape[0], 4), dtype=np.float32)
                pts_arr[:, 0:3] = xyz
                pts_arr[:, 3] = intensity
            elif arr.ndim == 1:
                try:
                    lst = [tuple(el) for el in arr]
                except Exception:
                    lst = []
                tmp = []
                for el in lst:
                    if len(el) >= 3:
                        x, y, z = float(el[0]), float(el[1]), float(el[2])
                        if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                            tmp.append((x / float(self._METERS_TO_UNREAL_UNIT), y / float(self._METERS_TO_UNREAL_UNIT), z / float(self._METERS_TO_UNREAL_UNIT)))
                if not tmp:
                    return None
                pts_arr = np.array(tmp, dtype=np.float32)
                intensity = np.linalg.norm(pts_arr[:, :3], axis=1).astype(np.float32)
                pts_arr = np.column_stack((pts_arr, intensity))
            else:
                return None

        elif isinstance(data, (list, tuple)):
            tmp = []
            for el in data:
                try:
                    if isinstance(el, (list, tuple)) and len(el) >= 3:
                        x, y, z = float(el[0]), float(el[1]), float(el[2])
                        if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                            tmp.append((x / float(self._METERS_TO_UNREAL_UNIT), y / float(self._METERS_TO_UNREAL_UNIT), z / float(self._METERS_TO_UNREAL_UNIT)))
                except Exception:
                    continue
            if not tmp:
                return None
            pts_arr = np.array(tmp, dtype=np.float32)
            intensity = np.linalg.norm(pts_arr[:, :3], axis=1).astype(np.float32)
            pts_arr = np.column_stack((pts_arr, intensity))
        else:
            return None

        if isinstance(pts_arr, list):
            pts_arr = np.array(pts_arr, dtype=np.float32)
        if pts_arr.ndim != 2 or pts_arr.shape[1] < 3:
            return None
        if pts_arr.shape[1] == 3:
            intensity = np.linalg.norm(pts_arr[:, :3], axis=1).astype(np.float32)
            pts_arr = np.column_stack((pts_arr, intensity))

        return pts_arr

    def _build_pointcloud2(self, pts_arr):
    
        pc = PointCloud2()
        h = Header()
        h.frame_id = getattr(self, 'scan_frame', '')
        if self._node is not None:
            try:
                h.stamp = self._node.get_clock().now().to_msg()
            except Exception:
                pass
        pc.header = h
        pc.height = 1
        pc.width = pts_arr.shape[0]

        # fields: x,y,z,intensity
        fields = []
        try:
            dtype_const = PointField.FLOAT32
        except Exception:
            dtype_const = 7
        offsets = [0, 4, 8, 12]
        names = ['x', 'y', 'z', 'intensity']
        for name, off in zip(names, offsets):
            pf = PointField()
            pf.name = name
            pf.offset = off
            pf.datatype = dtype_const
            pf.count = 1
            fields.append(pf)

        pc.fields = fields
        pc.is_bigendian = False
        pc.point_step = 16
        pc.row_step = pc.point_step * pc.width

        # pack data little-endian float32
        try:
            flat = pts_arr.astype(np.float32).flatten().tolist()
            pc.data = struct.pack('<' + ('f' * len(flat)), *flat)
        except Exception:
            return None

        pc.is_dense = True
        return pc


