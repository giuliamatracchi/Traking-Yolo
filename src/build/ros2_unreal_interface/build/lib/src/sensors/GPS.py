from .sensor import Sensor
import numpy as np
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Imu
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion, Twist, Vector3, TransformStamped
from tf2_ros import TransformBroadcaster




def change_ref_system(last_pose, current_pose, thr=1e-3):
    # calcolano una sola volta seno e coseno dei tre angoli di Eulero del last_pose
    cr, sr = np.cos(last_pose['roll']), np.sin(last_pose['roll'])
    cp, sp = np.cos(last_pose['pitch']), np.sin(last_pose['pitch'])
    cy, sy = np.cos(last_pose['yaw']), np.sin(last_pose['yaw'])

    #costruzione matrici elementari di rotazione
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)

    # matrice di rotazione complessiva associata a last_pose
    Rm = Rz @ Ry @ Rx

    
    t = np.array([last_pose['x'], last_pose['y'], last_pose['z']], dtype=float).reshape(3, 1)#posizione del last_pose nel frame di riferimento
    Tr = np.eye(4, dtype=float) # Matrice di trasformazione omogenea 4x4 
    Tr[:3, :3] = Rm
    Tr[:3, 3:4] = t

    # Punto corrente in coordinate omogenee
    p_cur = np.array([current_pose['x'], current_pose['y'], current_pose['z'], 1.0], dtype=float).reshape(4, 1)

    # calcola l’inversa della trasformazione omogenea Tr del “last_pose” per riportare la posizione corrente p_cur nel frame del last_pose.
    try:
        Tr_inv = np.linalg.inv(Tr) #Prova a invertire numericamente Tr con inv(Tr)
    except np.linalg.LinAlgError:
        # se l'inversone fallisce usa la formula analitica dell’inversa di una trasformazione rigida
        Rt = Rm.T
        Tr_inv = np.eye(4, dtype=float)
        Tr_inv[:3, :3] = Rt
        Tr_inv[:3, 3:4] = -Rt @ t

    relative_xyz = (Tr_inv @ p_cur).reshape(-1)[:3] #spostamento relativo

    # ripulisco dal rumore numerico
    # soglia sui piccoli valori
    relative_xyz = np.where(np.abs(relative_xyz) < thr, 0.0, relative_xyz)

    # Differenze angolari modulo 2*pi
    two_pi = 2.0 * np.pi
    d_roll = np.fmod(current_pose['roll'] - last_pose['roll'], two_pi)
    d_pitch = np.fmod(current_pose['pitch'] - last_pose['pitch'], two_pi)
    d_yaw = np.fmod(current_pose['yaw'] - last_pose['yaw'], two_pi)

    #output
    return {
        'x': float(relative_xyz[0]),
        'y': float(relative_xyz[1]),
        'z': float(relative_xyz[2]),
        'roll': float(d_roll),
        'pitch': float(d_pitch),
        'yaw': float(d_yaw),
    }



#metodo per convertire gli angoli in quaternioni
def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> Quaternion: #prende input in radianti e ritonra un tipo "Quaternione"
    cr2 = np.cos(roll * 0.5)
    sr2 = np.sin(roll * 0.5)
    cp2 = np.cos(pitch * 0.5)
    sp2 = np.sin(pitch * 0.5)
    cy2 = np.cos(yaw * 0.5)
    sy2 = np.sin(yaw * 0.5)

    qw = cr2 * cp2 * cy2 + sr2 * sp2 * sy2
    qx = sr2 * cp2 * cy2 - cr2 * sp2 * sy2
    qy = cr2 * sp2 * cy2 + sr2 * cp2 * sy2
    qz = cr2 * cp2 * sy2 - sr2 * sp2 * cy2
    q = Quaternion()
    q.x = float(qx)
    q.y = float(qy)
    q.z = float(qz)
    q.w = float(qw)
    return q


class GPS(Sensor):
    def __init__(self, node, env_topic, obs_settings, base_frame_default: str = 'base_link', topic_odom: str = "/odom", queue_size_odom: int = 10, topic_path: str = "/path", queue_size_path: int = 10, topic_imu: str = "/imu", queue_size_imu: int = 50, base_frame=None, odom_frame='odom', imu_frame='imu', publish_odom: bool = True, publish_path: bool = True, publish_imu: bool = True, tf_broadcast: bool = True, distance_thr: float = 0.1, thr: float = 1e-3, **kwargs,):
        super().__init__()

   
        self._node = node
        self._env_topic = env_topic
        self._obs_settings = obs_settings or {}
        self._unreal_settings = kwargs.get('unreal_settings', {}) or {}
        self._ue = kwargs.get('environment_ue')
        self._ue_sensor = kwargs.get('ue_sensor')
        self._sensor_type = kwargs.get('sensor_type', 'GPS')
        self._specific_name = kwargs.get('specific_name', 'default')

        #"sanificazione del nome"--< toglie spazi e simboli
        _san_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in str(self._specific_name)).lower()

        # Alias deprecato: permettiamo ancora 'camera_frame' via kwargs per retro-compatibilità
        camera_frame_kw = kwargs.get('camera_frame', None)
        if camera_frame_kw is not None:
            try:
                self._node.get_logger().warn("GPS.__init__: 'camera_frame' è deprecato, usa 'base_frame' o 'base_frame_default'.")
            except Exception:
                pass

        default_base = f"base_{_san_name}" if _san_name else (base_frame or base_frame_default or camera_frame_kw) # se c'è un nome specifico del sensore crea il frame dedicato
        default_imu = f"imu_{_san_name}" if _san_name else imu_frame
        self._base_frame = str(kwargs.get('base_frame', base_frame if base_frame is not None else default_base))
        self._odom_frame = str(kwargs.get('odom_frame', odom_frame))
        self._imu_frame = str(kwargs.get('imu_frame', imu_frame if imu_frame else default_imu))

        # Flags pubblicazione e TF
        self._publish_odom = bool(kwargs.get('publish_odom', publish_odom))
        self._publish_path = bool(kwargs.get('publish_path', publish_path))
        self._publish_imu = bool(kwargs.get('publish_imu', publish_imu))
        self._tf_broadcast = bool(kwargs.get('tf_broadcast', tf_broadcast))

        # Soglie e stato
        self._distance_thr = float(kwargs.get('distance_thr', distance_thr))
        self._thr = float(kwargs.get('thr', thr))
        self._current_time = None
        self._last_time = None
        self._last_pose = None
        self._last_velocity = None
        self._path_msg = None

        # Conversioni Unreal <-> metri
        meters_to_unreal = self._unreal_settings.get('METERS_TO_UNREAL_UNIT')
        if meters_to_unreal is None:
            meters_to_unreal = self._unreal_settings.get('__METERS_TO_UNREAL_UNIT')
        try:
            self._meters_to_unreal = float(meters_to_unreal) if meters_to_unreal is not None else 100.0
        except (TypeError, ValueError):
            self._meters_to_unreal = 100.0
        self._unreal_to_meters = (1.0 / self._meters_to_unreal) if self._meters_to_unreal > 0 else 0.01

        # Questo sensore si aspetta dati dall'ambiente (pose, velocità, ecc.)
        self._expects_data = True

        # Helper per comporre topic senza slash iniziale
        def _join_topic(*parts: str) -> str:
            cleaned = [str(p).strip().strip('/') for p in parts if p is not None and str(p).strip() != ""]
            return "/".join(cleaned)

        # Namespace base: /<env_topic>/<sensor_type>/<specific_name>
        base_topic = _join_topic(self._env_topic, self._sensor_type, self._specific_name)

        # Publisher: Odometry, Path, IMU
        odom_topic = _join_topic(base_topic, str(topic_odom))
        path_topic = _join_topic(base_topic, str(topic_path))
        imu_topic = _join_topic(base_topic, str(topic_imu))

        self._pub_odom = self._node.create_publisher(Odometry, odom_topic, int(queue_size_odom)) if self._publish_odom else None
        self._pub_path = self._node.create_publisher(Path, path_topic, int(queue_size_path)) if self._publish_path else None
        self._pub_imu = self._node.create_publisher(Imu, imu_topic, int(queue_size_imu)) if self._publish_imu else None

        # TF Broadcaster (tf2) opzionale
        self._tf_broadcaster = None
        if self._tf_broadcast and TransformBroadcaster is not None:
            try:
                self._tf_broadcaster = TransformBroadcaster(self._node)
            except Exception:
                self._tf_broadcaster = None

        # Pre-alloca header base (ri-aggiorneremo timestamp ad ogni publish)
        self._base_header = Header()
        self._base_header.frame_id = self._base_frame

    def change_settings(self):
        pass


    def publish_observation(self, data=None):
        # Atteso: data come sequenza con pos (6,7,8) in Unreal units e angoli (9,10,11) in gradi [pitch, yaw, roll]
        if data is None:
            return

        # Tempo attuale (rclpy Time) e delta t
        now_time = self._node.get_clock().now()

        # Estrai e converti pose corrente
        try:
            pitch = float(np.deg2rad(data[9]))
            yaw = float(np.deg2rad(data[10]))
            roll = float(np.deg2rad(data[11]))
            x = float(data[6]) * self._unreal_to_meters
            y = float(data[7]) * self._unreal_to_meters
            z = float(data[8]) * self._unreal_to_meters
        except Exception:
            return

        current = {'x': x, 'y': y, 'z': z, 'roll': roll, 'pitch': pitch, 'yaw': yaw}

        # Prima chiamata: inizializza stato e ritorna (un passo di ritardo per velocità)
        if self._current_time is None:
            self._current_time = now_time
            self._last_time = now_time
            self._last_pose = dict(current)
            return

        # Seconda chiamata: calcola velocità iniziali da posa relativa e ritorna (due passi per accelerazioni)
        if self._last_velocity is None:
            self._current_time = now_time
            dt = (self._current_time - self._last_time).nanoseconds * 1e-9
            if dt <= 0:
                dt = 1e-6
            relative = change_ref_system(self._last_pose, current, self._thr)
            last_vx = relative['x'] / dt
            last_vy = relative['y'] / dt
            last_vz = relative['z'] / dt
            last_vroll = relative['roll'] / dt
            last_vpitch = relative['pitch'] / dt
            last_vyaw = relative['yaw'] / dt
            self._last_time = self._current_time
            self._last_pose = dict(current)
            self._last_velocity = {'x': last_vx, 'y': last_vy, 'z': last_vz, 'roll': last_vroll, 'pitch': last_vpitch, 'yaw': last_vyaw}
            return

        # Aggiorna tempo e dt
        self._current_time = now_time
        dt = (self._current_time - self._last_time).nanoseconds * 1e-9
        if dt <= 0:
            dt = 1e-6

        # Quaternion orientazione corrente
        q = euler_to_quaternion(current['roll'], current['pitch'], current['yaw'])
        now_msg = self._current_time.to_msg()

        # TF odom->base opzionale
        if self._tf_broadcast and self._tf_broadcaster is not None:
            ts = TransformStamped()
            ts.header.stamp = now_msg
            ts.header.frame_id = self._odom_frame
            ts.child_frame_id = self._base_frame
            ts.transform.translation.x = current['x']
            ts.transform.translation.y = current['y']
            ts.transform.translation.z = current['z']
            ts.transform.rotation = q
            try:
                self._tf_broadcaster.sendTransform(ts)
            except Exception:
                pass

        # Path opzionale (accumulo con soglia distanza)
        if self._pub_path is not None:
            curr_ps = PoseStamped()
            curr_ps.header.stamp = now_msg
            curr_ps.header.frame_id = self._odom_frame
            curr_ps.pose = Pose(position=Point(x=current['x'], y=current['y'], z=current['z']), orientation=q)
            if self._path_msg is None:
                self._path_msg = [curr_ps]
            else:
                last_pose_ps = self._path_msg[-1].pose
                dx = current['x'] - last_pose_ps.position.x
                dy = current['y'] - last_pose_ps.position.y
                dz = current['z'] - last_pose_ps.position.z
                if (dx*dx + dy*dy + dz*dz) ** 0.5 > self._distance_thr:
                    self._path_msg.append(curr_ps)
            path = Path()
            path.header.stamp = now_msg
            path.header.frame_id = self._odom_frame
            path.poses = self._path_msg
            try:
                self._pub_path.publish(path)
            except Exception:
                pass

        # Odometry
        if self._pub_odom is not None:
            # Velocità semplici in world/odom
            vel_x = (current['x'] - self._last_pose['x']) / dt
            vel_y = (current['y'] - self._last_pose['y']) / dt
            vel_z = (current['z'] - self._last_pose['z']) / dt
            vel_roll = (current['roll'] - self._last_pose['roll']) / dt
            vel_pitch = (current['pitch'] - self._last_pose['pitch']) / dt
            vel_yaw = (current['yaw'] - self._last_pose['yaw']) / dt

            odom = Odometry()
            odom.header.stamp = now_msg
            odom.header.frame_id = self._odom_frame
            odom.child_frame_id = self._base_frame
            odom.pose.pose = Pose(position=Point(x=current['x'], y=current['y'], z=current['z']), orientation=q)
            diag = 0.017
            odom.pose.covariance = [diag,0.0,0.0,0.0,0.0,0.0,
                                    0.0,diag,0.0,0.0,0.0,0.0,
                                    0.0,0.0,diag,0.0,0.0,0.0,
                                    0.0,0.0,0.0,diag,0.0,0.0,
                                    0.0,0.0,0.0,0.0,diag,0.0,
                                    0.0,0.0,0.0,0.0,0.0,diag]
            odom.twist.twist = Twist(linear=Vector3(x=vel_x, y=vel_y, z=vel_z), angular=Vector3(x=vel_roll, y=vel_pitch, z=vel_yaw))
            odom.twist.covariance = [diag,0.0,0.0,0.0,0.0,0.0,
                                     0.0,diag,0.0,0.0,0.0,0.0,
                                     0.0,0.0,diag,0.0,0.0,0.0,
                                     0.0,0.0,0.0,diag,0.0,0.0,
                                     0.0,0.0,0.0,0.0,diag,0.0,
                                     0.0,0.0,0.0,0.0,0.0,diag]
            try:
                self._pub_odom.publish(odom)
            except Exception:
                pass

        # IMU
        if self._pub_imu is not None:
            relative = change_ref_system(self._last_pose, current, self._thr)
            vx = relative['x'] / dt
            vy = relative['y'] / dt
            vz = relative['z'] / dt
            v_roll = relative['roll'] / dt
            v_pitch = relative['pitch'] / dt
            v_yaw = relative['yaw'] / dt
            current_velocity = {'x': vx, 'y': vy, 'z': vz, 'roll': v_roll, 'pitch': v_pitch, 'yaw': v_yaw}

            # Accelerazioni puramente cinematiche (nessuna correzione di gravità)
            acc_x = (current_velocity['x'] - self._last_velocity['x']) / dt
            acc_y = (current_velocity['y'] - self._last_velocity['y']) / dt
            acc_z = (current_velocity['z'] - self._last_velocity['z']) / dt

            imu = Imu()
            imu.header.stamp = now_msg
            imu.header.frame_id = self._imu_frame
            imu.orientation = q
            imu.angular_velocity = Vector3(x=current_velocity['roll'], y=current_velocity['pitch'], z=current_velocity['yaw'])
            imu.linear_acceleration = Vector3(x=acc_x, y=acc_y, z=acc_z)
            try:
                self._pub_imu.publish(imu)
            except Exception:
                pass
            self._last_velocity = dict(current_velocity)

        # Aggiorna stato per il prossimo step
        self._last_time = self._current_time
        self._last_pose = dict(current)