import yaml
import rclpy
from rclpy.node import Node
import threading
import time
import os
import importlib
import numpy as np
import json

from syndatatoolbox_api.environment import Environment as UEEnvironment
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry


class Environment(Node):
    def __init__(self):
        super().__init__('environment')

        # Connessione
        yaml_file = os.path.join(os.path.dirname(__file__), "setup", "settings.yaml")
        params_from_yaml = {}

        if os.path.exists(yaml_file):
            try:
                with open(yaml_file, "r") as f:
                    data = yaml.safe_load(f) or {}
                params_from_yaml = data.get("environment", {}).get("ros__parameters", {})
                self.get_logger().info(f"Caricati parametri da {yaml_file}")
            except Exception as e:
                self.get_logger().error(f"Errore nel caricamento di settings.yaml: {e}")

        # Parametri
        self.declare_parameter('address', params_from_yaml.get('address', ''))
        self.declare_parameter('port', params_from_yaml.get('port', 0))
        self.declare_parameter('fps', 10.0)
        self.declare_parameter('motion_speed_mps', 0.5)
        self.declare_parameter('position_tolerance_m', 0.05)

        # Lettura parametri
        address = self.get_parameter('address').value
        port = int(self.get_parameter('port').value)
        self.fps = float(self.get_parameter('fps').value)
        self.motion_speed_mps = float(self.get_parameter('motion_speed_mps').value)
        self.position_tolerance_m = float(self.get_parameter('position_tolerance_m').value)

        setup = params_from_yaml.get("setup", {})
        if not isinstance(setup, dict):
            setup = {}

        # Check parametri
        if not address or port == 0:
            raise RuntimeError(
                "Parametri 'address' e 'port' mancanti. "
                "Assicurati che settings.yaml sia presente o passa --params-file."
            )

        # Check import SDK
        if UEEnvironment is None:
            raise RuntimeError(
                "syndatatoolbox_api non disponibile: installa e verifica il PYTHONPATH dell'SDK."
            )

        self.ue_env = UEEnvironment(port=port, address=address, setup=setup)
        self.get_logger().info(f"Connessione UE5 stabilita: {address}:{port}")

        # Sensori disponibili
        self.sensor_used = list(self.ue_env.sensor_set.keys()) if hasattr(self.ue_env, 'sensor_set') else []
        self.get_logger().info(f"Sensori UE5 disponibili: {self.sensor_used}")

        self.sensors = {}
        self._sensor_lock = threading.Lock()
        self._sensor_thread_stop = threading.Event()
        self._ue_failures = 0
        self._specific_names = []

        # Costruzione sensori e avvio thread
        self.build_sensors()
        self._sensor_thread = threading.Thread(
            target=self.publish_loop,
            name='sensors_thread',
            daemon=True
        )
        self._sensor_thread.start()

        # Action manager
        self.action_manager = "CoordinateActionManager(CoordinateActionManagerSDT)"
        self.action_type = "MOVETO"

        # Target desiderato
        self.des_pose = None
        self.des_orientation = None


        # Stato reale dell'ambulanza da odometria
        self.current_real_pose = None

        # Timer/action timing
        self.last_action_time = time.perf_counter()

        # Subscriber target ambulanza
        self.action_sub = self.create_subscription(
            PoseStamped,
            "/ambulance_position_des",
            self.update_des_pose,
            10
        )

        # Subscriber odometria reale ambulanza
        self.odom_sub = self.create_subscription(
            Odometry,
            "/environment/GPS/GPSSDTAMBULANCE/odom",
            self.odom_callback,
            20
        )

        # Timer action manager
        self.action_manager_timer = self.create_timer(
            0.1,
            self.action_manager_callback
        )

        # Arresta il thread sensori allo shutdown
        try:
            ctx = rclpy.get_default_context()
            if hasattr(ctx, 'on_shutdown') and callable(getattr(ctx, 'on_shutdown')):
                ctx.on_shutdown(self._sensor_thread_stop.set)
            else:
                self.get_logger().warning(
                    "on_shutdown non disponibile nel contesto rclpy; userò il fallback in destroy_node."
                )
        except Exception as e:
            self.get_logger().warning(f"Impossibile registrare on_shutdown: {e}")

    def update_des_pose(self, data: PoseStamped):
        self.des_pose = np.array(
            [
                data.pose.position.x,
                data.pose.position.y,
                data.pose.position.z
            ],
            dtype=float
        )

        self.des_orientation = np.array(
            [
                data.pose.orientation.x,
                data.pose.orientation.y,
                data.pose.orientation.z,
                data.pose.orientation.w
            ],
            dtype=float
        )

    def odom_callback(self, msg: Odometry):
        self.current_real_pose = np.array(
            [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z
            ],
            dtype=float
        )

    def action_manager_callback(self):
        if self.des_pose is None or self.des_orientation is None:
            return

        if self.current_real_pose is None:
            return

        now = time.perf_counter()
        dt = max(1e-3, now - self.last_action_time)
        self.last_action_time = now

        # Calcolo sempre dal punto reale attuale dell'ambulanza
        delta = self.des_pose - self.current_real_pose
        dist = np.linalg.norm(delta)

        if dist <= self.position_tolerance_m:
            cmd_pose = self.des_pose.copy()
        else:
            max_step = self.motion_speed_mps * dt
            step = min(max_step, dist)
            direction = delta / dist
            cmd_pose = self.current_real_pose + direction * step

        cmd_orientation = self.des_orientation.copy()

        action = list(np.concatenate([cmd_pose, cmd_orientation]))
        self.ue_env.perform_action(self.action_manager, {self.action_type: action})


    def build_sensors(self, sensors_used=None, settings_dir=None, env_topic=None):
        if sensors_used is None:
            sensors_used = self.sensor_used
        if settings_dir is None:
            settings_dir = os.path.join(os.path.dirname(__file__), 'settings')
        if env_topic is None:
            env_topic = self.get_name()

        built = {}
        ok_specific_names = []
        ok_ue_names = []

        for obs in sensors_used or []:
            if '(' in obs and ')' in obs:
                sensor_type = obs.split('(')[0].strip()
                specific_name = obs.split('(')[1].split(')')[0].strip()
            else:
                sensor_type = obs.strip()
                specific_name = obs.strip()

            ue_sensor = self.ue_env.sensor_set.get(obs)
            obs_settings = getattr(ue_sensor, 'settings', {}) if ue_sensor is not None else {}

            try:
                base_pkg = __package__
                if base_pkg:
                    module_name = f"{base_pkg}.sensors.{sensor_type}"
                else:
                    module_name = f"sensors.{sensor_type}"
                module = importlib.import_module(module_name)
            except ModuleNotFoundError:
                self.get_logger().warning(
                    f"Modulo per sensore '{sensor_type}' non trovato. Salto '{obs}'."
                )
                continue

            try:
                class_ = getattr(module, sensor_type)
            except AttributeError:
                self.get_logger().warning(
                    f"Classe sensore '{sensor_type}' non implementata nel modulo. Salto '{obs}'."
                )
                continue

            json_path = os.path.join(settings_dir, f'{sensor_type}.json')
            settings = {}

            try:
                with open(json_path, 'r') as f:
                    settings = json.load(f) or {}
            except FileNotFoundError:
                self.get_logger().warning(
                    f"File di configurazione '{json_path}' non trovato per '{sensor_type}'. Uso impostazioni vuote."
                )

            json_path_unreal = os.path.join(settings_dir, 'unreal_settings.json')
            if os.path.exists(json_path_unreal):
                with open(json_path_unreal, 'r') as f_unreal:
                    settings['unreal_settings'] = json.load(f_unreal) or {}

            settings['env_topic'] = env_topic
            settings['obs_settings'] = obs_settings
            settings['node'] = self
            settings['environment_ue'] = self.ue_env
            settings['ue_sensor'] = ue_sensor
            settings['sensor_type'] = sensor_type
            settings['specific_name'] = specific_name

            sanitized_specific_name = ''.join(
                c if c.isalnum() or c == '_' else '_' for c in specific_name
            ).lower()

            if sensor_type in ('RGBCamera', 'DepthCamera', 'SegmentationCamera'):
                settings['camera_frame'] = f"camera_{sanitized_specific_name}"
            elif sensor_type == 'GPS':
                settings['base_frame_default'] = f"base_{sanitized_specific_name}"

            try:
                instance = class_(**settings)
            except Exception as e:
                self.get_logger().warning(
                    f"Impossibile istanziare sensore '{sensor_type}' ('{specific_name}'): {e}. Salto."
                )
                continue

            built[specific_name] = instance
            ok_specific_names.append(specific_name)
            ok_ue_names.append(obs)
            self.get_logger().info(f"Creato sensore {sensor_type} -> '{specific_name}'")

        with self._sensor_lock:
            self.sensors = built

        self._specific_names = ok_specific_names
        self.sensor_used = ok_ue_names
        self.get_logger().info(f"Sensori costruiti: {list(self.sensors.keys())}")
        return built

    def publish_loop(self):
        period = 1.0 / self.fps if self.fps and self.fps > 0 else 0.0
        next_time = time.perf_counter()

        while not self._sensor_thread_stop.is_set():
            if not self.sensor_used:
                if period > 0:
                    next_time += period
                    sleep_time = max(0.0, next_time - time.perf_counter())
                    if sleep_time > 0:
                        self._sensor_thread_stop.wait(timeout=sleep_time)
                else:
                    self._sensor_thread_stop.wait(timeout=0.1)
                continue

            try:
                observations = self.ue_env.get_obs(self.sensor_used)
                if self._ue_failures > 0:
                    self.get_logger().info("Connessione UE ripristinata. Riprendo la pubblicazione.")
                self._ue_failures = 0
            except Exception as e:
                self._ue_failures += 1
                if self._ue_failures == 1:
                    self.get_logger().error(
                        f"Errore UE get_obs: {e}. Connessione persa? Ritento con backoff."
                    )
                elif self._ue_failures % 10 == 0:
                    self.get_logger().warning(
                        f"UE disconnesso da {self._ue_failures} tentativi: {e}"
                    )

                backoff = min(1.0 * (2 ** (self._ue_failures - 1)), 10.0)
                self._sensor_thread_stop.wait(timeout=backoff)
                continue

            if isinstance(observations, (list, tuple)):
                if len(observations) != len(self.sensor_used):
                    self.get_logger().warning(
                        f"Dimensione osservazioni ({len(observations)}) diversa da sensor_used ({len(self.sensor_used)}). Uso zip parziale."
                    )
                observations = {ue: val for ue, val in zip(self.sensor_used, observations)}
            elif observations is not None and not isinstance(observations, dict) and len(self.sensor_used) == 1:
                observations = {self.sensor_used[0]: observations}

            if observations:
                try:
                    if isinstance(observations, dict):
                        with self._sensor_lock:
                            for ue_name, specific_name in zip(self.sensor_used, self._specific_names):
                                sensor = self.sensors.get(specific_name)
                                if sensor is None:
                                    continue

                                data = observations.get(ue_name, observations.get(specific_name))

                                if data is None:
                                    if not getattr(sensor, '_expects_data', False):
                                        try:
                                            sensor.publish_observation()
                                        except Exception as e:
                                            self.get_logger().error(
                                                f"Errore pubblicando {specific_name}: {e}"
                                            )
                                    continue

                                if getattr(sensor, '_expects_data', False):
                                    try:
                                        sensor.publish_observation(data)
                                    except Exception as e:
                                        self.get_logger().error(
                                            f"Errore pubblicando {specific_name}: {e}"
                                        )
                                else:
                                    try:
                                        sensor.publish_observation()
                                    except Exception as e:
                                        self.get_logger().error(
                                            f"Errore pubblicando {specific_name}: {e}"
                                        )
                except Exception as e:
                    self.get_logger().error(f"Errore durante la pubblicazione: {e}")

            if period > 0:
                next_time += period
                sleep_time = max(0.0, next_time - time.perf_counter())
                if sleep_time > 0:
                    self._sensor_thread_stop.wait(timeout=sleep_time)
            else:
                self._sensor_thread_stop.wait(timeout=0.01)

    def destroy_node(self):
        try:
            self._sensor_thread_stop.set()
            if hasattr(self, '_sensor_thread') and self._sensor_thread is not None:
                self._sensor_thread.join(timeout=2.0)
        except Exception:
            pass
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Environment()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

