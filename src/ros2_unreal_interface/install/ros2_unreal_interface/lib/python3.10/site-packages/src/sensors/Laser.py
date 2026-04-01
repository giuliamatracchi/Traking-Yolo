from .sensor import Sensor
import math
import numpy as np
from std_msgs.msg import Header
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data, QoSProfile
from copy import deepcopy


class Laser(Sensor):
    def __init__(
        self, node, env_topic, obs_settings, scan_frame: str = 'base_link', topic_scan: str = '/scan', publish: bool = True, queue_size: int = 1,**kwargs,):
        super().__init__()

        self._node = node
        self._env_topic = env_topic
        self._obs_settings = obs_settings or {}
        self._unreal_settings = kwargs.get('unreal_settings', {}) or {}
        self._sensor_type = kwargs.get('sensor_type', 'Laser')
        self._specific_name = kwargs.get('specific_name', 'default')

        # Unreal units -> metres (default UE uses cm => 100)
        self._METERS_TO_UNREAL_UNIT = float(self._unreal_settings.get('METERS_TO_UNREAL_UNIT', self._unreal_settings.get('__METERS_TO_UNREAL_UNIT', 100.0)))

      
        self.scan_frame = kwargs.get('scan_frame', scan_frame) or scan_frame
        self._topic_scan = kwargs.get('scan_topic', topic_scan) or topic_scan

        # pubblicazione: preferenza ordine -> kwargs (file JSON passato come kwargs),
        # poi il parametro esplicito 'publish' della funzione, poi obs_settings, poi default.
        if 'publish' in kwargs:
            _p = kwargs.get('publish')
        else:
            _p = publish if publish is not None else self._obs_settings.get('publish', False)
        self._publish = bool(_p)

        # queue_size: prefer kwargs, poi parametro esplicito, poi obs_settings, poi default
        if 'queue_size' in kwargs:
            _q = kwargs.get('queue_size')
        else:
            _q = queue_size if queue_size is not None else self._obs_settings.get('queue_size', 1)
        try:
            self._queue_size = int(_q)
        except Exception:
            self._queue_size = 1

        self._expects_data = True

        #costruzione topic su cui viene pubblicato il laser scan
        ns_sensor = f"/{str(self._env_topic).lstrip('/')}/{str(self._sensor_type).strip()}"
        ns_specific = f"{ns_sensor}/{str(self._specific_name).strip()}"
        scan_topic_full = f"{ns_specific}/{str(self._topic_scan).lstrip('/')}"

        # early diagnostic: report what the constructor saw for publish/queue/node
        try:
            self._node.get_logger().info(
                f"Laser.__init__: publish_in_kwargs={'publish' in kwargs}, publish_resolved={self._publish}, queue_size={self._queue_size}, node_present={self._node is not None}, topic={scan_topic_full}"
            )
        except Exception:
            pass

        self._pub_scan = None
        if self._publish and self._node is not None:
            # preferiamo usare qos_profile_sensor_data; se non disponibile usiamo QoSProfile(depth=...)
            try:
                qos = qos_profile_sensor_data
            except Exception:
                qos = QoSProfile(depth=self._queue_size)

            try:
                self._pub_scan = self._node.create_publisher(LaserScan, scan_topic_full, qos)
            except Exception as e:
                try:
                    self._node.get_logger().warning(f"Laser: create_publisher failed for {scan_topic_full}: {e}")
                except Exception:
                    pass
                self._pub_scan = None

        # diagnostic logging so we can see at runtime whether publishing was enabled
        try:
            if self._publish and self._pub_scan is not None:
                self._node.get_logger().info(f"Laser: publisher created for {scan_topic_full}")
            elif self._publish and self._pub_scan is None:
                self._node.get_logger().warning(f"Laser: publishing requested but publisher not created for {scan_topic_full}")
            else:
                # publishing disabled by config
                try:
                    self._node.get_logger().debug(f"Laser: publishing disabled for {scan_topic_full}")
                except Exception:
                    pass
        except Exception:
            # keep constructor robust if logging fails
            pass


        # default LaserScan message
        self._scan_msg = LaserScan()
        self._scan_msg.header = Header()
        self._scan_msg.header.frame_id = self.scan_frame

        def _g(k, default):
            v = kwargs.get(k) if k in kwargs else self._obs_settings.get(k)
            return v if v is not None else default

        # valori di base (falliscono silenziosamente solo durante conversione)
        try:
            self._scan_msg.angle_min = float(_g('angle_min', -math.pi))
            self._scan_msg.angle_max = float(_g('angle_max', math.pi))
            self._scan_msg.angle_increment = float(_g('angle_increment', 0.0))
            self._scan_msg.time_increment = float(_g('time_increment', 0.0))
            self._scan_msg.scan_time = float(_g('scan_time', 0.0))
            self._scan_msg.range_min = float(_g('range_min', 0.0))
            self._scan_msg.range_max = float(_g('range_max', 100.0))
        except Exception:
            # se qualcosa non è convertibile, manteniamo i default
            pass

        # proviamo a costruire un LaserScan coerente dalle obs_settings
        try:
            self._scan_msg = self._build_laser_scan_msg(self._obs_settings or {})
        except Exception:
            # manteniamo i default precedenti
            pass

        self._xy = [[], []]

    def change_settings(self, **kwargs):
        # runtime changes non supportate per ora
        pass

    def _build_laser_scan_msg(self, obs_settings):
        h = Header()
        if self._node is not None:
            try:
                h.stamp = self._node.get_clock().now().to_msg()
            except Exception:
                # ok se non disponibile
                pass
        h.frame_id = getattr(self, 'scan_frame', '')

        laser_scan_msg = LaserScan()
        laser_scan_msg.header = h

        def _deg_to_rad(v, default):
            try:
                return float(np.deg2rad(float(v)))
            except Exception:
                return default

        angle_min = _deg_to_rad(obs_settings.get('start_angle_x'), getattr(self._scan_msg, 'angle_min', -math.pi))
        angle_max = _deg_to_rad(obs_settings.get('end_angle_x'), getattr(self._scan_msg, 'angle_max', math.pi))
        angle_inc = _deg_to_rad(obs_settings.get('distance_angle_x'), getattr(self._scan_msg, 'angle_increment', 0.0))

        laser_scan_msg.angle_min = float(angle_min)
        laser_scan_msg.angle_max = float(angle_max)
        laser_scan_msg.angle_increment = float(angle_inc)
        laser_scan_msg.time_increment = 0.0
        laser_scan_msg.range_min = float(getattr(self._scan_msg, 'range_min', 0.0))

        lr = obs_settings.get('laser_range')
        if lr is not None:
            try:
                laser_scan_msg.range_max = float(lr) / float(self._METERS_TO_UNREAL_UNIT)
            except Exception:
                laser_scan_msg.range_max = float(getattr(self._scan_msg, 'range_max', 100.0))
        else:
            laser_scan_msg.range_max = float(getattr(self._scan_msg, 'range_max', 100.0))

        laser_scan_msg.ranges = []
        laser_scan_msg.intensities = []
        return laser_scan_msg

    def publish_observation(self, data):
        if data is None:
            return

        # estrai sequenza da dict
        if isinstance(data, dict):
            for k in ('points', 'xy', 'data'):
                if k in data:
                    data = data[k]
                    break

        pts = []
        intensities_raw = []

        # numpy-like
        if hasattr(data, 'shape') and hasattr(data, 'dtype'):
            arr = np.asarray(data)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                # (N,2) or (N,>=3)
                pts = [(float(x), float(y)) for x, y in arr[:, :2]]
                if arr.shape[1] >= 3:
                    intensities_raw = [float(v) for v in arr[:, 2].tolist()]
            elif arr.ndim == 1:
                # array di tuple/struct
                try:
                    for el in arr:
                        if len(el) >= 2:
                            pts.append((float(el[0]), float(el[1])))
                            if len(el) >= 3:
                                intensities_raw.append(float(el[2]))
                except Exception:
                    pts = []
            else:
                # possibile immagine di profondità: estrai riga centrale
                try:
                    h_idx = arr.shape[0] // 2
                    center_row = np.asarray(arr[h_idx, :]).flatten()
                    if center_row.size > 0:
                        maxv = float(np.nanmax(center_row))
                        if maxv <= 1.1:
                            # normalized -> proviamo a scalare con laser_range se presente
                            laser_range_setting = None
                            try:
                                laser_range_setting = float(self._obs_settings.get('laser_range', 0))
                            except Exception:
                                laser_range_setting = None
                            if laser_range_setting:
                                scale = float(laser_range_setting) / float(self._METERS_TO_UNREAL_UNIT)
                                meters_list = [float(v) * scale for v in center_row]
                            else:
                                meters_list = [float(v) for v in center_row]
                        elif maxv <= 100.0:
                            meters_list = [float(v) for v in center_row]
                        else:
                            meters_list = [float(v) / float(self._METERS_TO_UNREAL_UNIT) for v in center_row]
                        pts = [(r, 0.0) for r in meters_list]
                except Exception:
                    pts = []

        # lista/tuple
        elif isinstance(data, (list, tuple)):
            if len(data) == 0:
                pts = []
            elif all(isinstance(el, (list, tuple)) and len(el) >= 2 for el in data):
                for el in data:
                    try:
                        pts.append((float(el[0]), float(el[1])))
                        if len(el) >= 3:
                            intensities_raw.append(float(el[2]))
                    except Exception:
                        continue
            elif all(isinstance(el, (int, float)) for el in data) and len(data) % 2 == 0:
                it = iter(data)
                pts = [(float(x), float(y)) for x, y in zip(it, it)]
            else:
                # fallback generico
                try:
                    for el in data:
                        pts.append((float(el[0]), float(el[1])))
                except Exception:
                    pts = []

        else:
            # tipo non supportato
            pts = []

        if not pts:
            return

        # convert UE units (cm) -> metri quando serve (valori grandi >10 indicano cm)
        list_x = []
        list_y = []
        ranges = []
        for x_v, y_v in pts:
            try:
                raw_x = float(x_v)
                raw_y = float(y_v)
            except Exception:
                continue
            xm = raw_x / self._METERS_TO_UNREAL_UNIT if abs(raw_x) > 10.0 else raw_x
            ym = raw_y / self._METERS_TO_UNREAL_UNIT if abs(raw_y) > 10.0 else raw_y
            list_x.append(xm)
            list_y.append(ym)
            ranges.append(float(math.hypot(xm, ym)))

        self._xy = [list_x, list_y]

        # prepara il messaggio e normalizza ranges/intensities
        # usa una copia del template per evitare side-effect quando il messaggio
        # viene modificato e pubblicato ripetutamente
        msg = deepcopy(self._scan_msg)
        if self._node is not None:
            try:
                msg.header.stamp = self._node.get_clock().now().to_msg()
            except Exception:
                pass
        msg.header.frame_id = getattr(self, 'scan_frame', msg.header.frame_id)

        # se angle_increment non impostato e abbiamo più sample, calcoliamolo
        if (not msg.angle_increment or float(msg.angle_increment) == 0.0) and len(ranges) > 1:
            span = float(msg.angle_max) - float(msg.angle_min)
            msg.angle_increment = span / max(1, (len(ranges) - 1))

        # qualsiasi valore <=0 o non-finite -> inf
        norm = []
        for r in ranges:
            rv = float(r)
            if (not math.isfinite(rv)) or rv <= 0.0:
                norm.append(math.inf)
            else:
                norm.append(rv)
        ranges = norm

        # allineamento lunghezza rispetto alle impostazioni angolari
        if msg.angle_increment and float(msg.angle_increment) > 0.0:
            span = float(msg.angle_max) - float(msg.angle_min)
            expected_n = int(round(span / float(msg.angle_increment))) + 1
            if len(ranges) > expected_n:
                ranges = ranges[:expected_n]
            elif len(ranges) < expected_n:
                ranges = ranges + [math.inf] * (expected_n - len(ranges))

        # intensities: se disponibili normalizziamo, altrimenti usiamo proxy da range
        ints = []
        if intensities_raw:
            for v in intensities_raw:
                try:
                    vv = float(v)
                    ints.append(vv if math.isfinite(vv) else 0.0)
                except Exception:
                    ints.append(0.0)
            # align lengths
            if len(ints) > len(ranges):
                ints = ints[:len(ranges)]
            elif len(ints) < len(ranges):
                ints = ints + [0.0] * (len(ranges) - len(ints))

            # azzeriamo le intensity per range inf
            for i, r in enumerate(ranges):
                if not math.isfinite(r):
                    ints[i] = 0.0

            # normalizzazione min-max
            if ints:
                min_v = min(ints)
                max_v = max(ints)
                if abs(max_v - min_v) < 1e-9:
                    if abs(max_v) < 1e-9:
                        # fallback proxy
                        rmax = float(getattr(msg, 'range_max', 100.0)) or 100.0
                        msg.intensities = [0.0 if not math.isfinite(r) else 1.0 - min(max(r / rmax, 0.0), 1.0) for r in ranges]
                    else:
                        msg.intensities = [1.0 if v > 0.0 else 0.0 for v in ints]
                else:
                    span = float(max_v - min_v)
                    msg.intensities = [max(0.0, min(1.0, (float(v) - min_v) / span)) for v in ints]
            else:
                msg.intensities = [0.0] * len(ranges)
        else:
            # proxy intensities dal range
            rmax = float(getattr(msg, 'range_max', 100.0)) or 100.0
            proxy = [0.0 if not math.isfinite(r) else 1.0 - min(max(r / rmax, 0.0), 1.0) for r in ranges]
            msg.intensities = proxy

        msg.ranges = ranges

        if self._pub_scan is not None:
            try:
                self._pub_scan.publish(msg)
            except Exception:
                # non blocchiamo se la publish fallisce
                pass
