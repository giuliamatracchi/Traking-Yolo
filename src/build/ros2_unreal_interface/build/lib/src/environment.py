import yaml
import rclpy, sys
from rclpy.node import Node
import threading
import time
import os
import importlib
import numpy as np
import json
from syndatatoolbox_api.environment import Environment as UEEnvironment
from geometry_msgs.msg import PoseStamped


class Environment(Node):

 

    def __init__(self):
        super().__init__('environment')
        
        #connessione
        yaml_file = os.path.join(os.path.dirname(__file__), "setup", "settings.yaml")
        params_from_yaml = {} #dizionario vuoto per contenere i parametri letti
        if os.path.exists(yaml_file): #controlla se il file yamel esiste, in caso affermativo lo apre e carica i pametri
            try:
                with open(yaml_file, "r") as f:
                    data = yaml.safe_load(f) or {}
                params_from_yaml = data.get("environment", {}).get("ros__parameters", {})
                self.get_logger().info(f"Caricati parametri da {yaml_file}")
            except Exception as e:
                self.get_logger().error(f"Errore nel caricamento di settings.yaml: {e}")
        
        #dichiarazione dei parametri per il nodo, vengono presi di default dal file yamel
        self.declare_parameter('address', params_from_yaml.get('address', ''))
        self.declare_parameter('port', params_from_yaml.get('port', 0))
        self.declare_parameter('fps', 10.0)  # fps resta nel codice
    
        

        #lettura parametri
        address = self.get_parameter('address').value
        port = int(self.get_parameter('port').value)
        self.fps = float(self.get_parameter('fps').value)
        setup = params_from_yaml.get("setup", {}) if isinstance(params_from_yaml.get("setup", {}), dict) else {} #legge la sottoezione setup, si assicura che sia un dict, altrimenti usa un dict vuoto

        #check per controllo parametri mancanti
        if not address or port == 0:
            raise RuntimeError(
                "Parametri 'address' e 'port' mancanti. " 
                "Assicurati che settings.yaml sia presente o passa --params-file."
            )

        #check dell'import 
        if UEEnvironment is None:
            raise RuntimeError('syndatatoolbox_api non disponibile: installa e verifica il PYTHONPATH dell\'SDK.')
        self.ue_env = UEEnvironment(port=port, address=address, setup=setup) #crea un'instanza di environment con i parametri giusti
        self.get_logger().info(f"Connessione UE5 stabilita: {address}:{port}")


        # ottengo la lista dei sensori disponibili da UE usando sensor_set
        self.sensor_used = list(self.ue_env.sensor_set.keys()) if hasattr(self.ue_env, 'sensor_set') else []
        self.get_logger().info(f"Sensori UE5 disponibili: {self.sensor_used}")

        self.sensors = {}
        self._sensor_lock = threading.Lock()
        self._sensor_thread_stop = threading.Event()
        self._ue_failures = 0  # consecutive UE get_obs failures for backoff/log throttling

        # nomi specifici tra graffe--<in build viene riempita
        self._specific_names = []

        # Costruzione sensori e avvio thread
        self.build_sensors() 
        self._sensor_thread = threading.Thread(target=self.publish_loop, name='sensors_thread', daemon=True)
        self._sensor_thread.start()

        self.action_manager = "CoordinateActionManager(CoordinateActionManagerSDT)"
        self.action_type = "MOVETO"
        self.des_pose = None
        self.des_orientation = None
        self.action_sub = self.create_subscription(PoseStamped, "/ambulance_position_des", self.update_des_pose, 10)
        self.action_manager_timer = self.create_timer(0.1, self.action_manager_callback)
        
        # Arresta il thread sensori quando parte rclpy.shutdown()
        try:
            ctx = rclpy.get_default_context()
            if hasattr(ctx, 'on_shutdown') and callable(getattr(ctx, 'on_shutdown')):
                ctx.on_shutdown(self._sensor_thread_stop.set)
            else:
                self.get_logger().warning("on_shutdown non disponibile nel contesto rclpy; userò il fallback in destroy_node.")
        except Exception as e:
            # Fallback: se non riesce a registrare il callback, logga ma continua
            self.get_logger().warning(f"Impossibile registrare on_shutdown: {e}")
      

    def update_des_pose(self, data):
        self.des_pose = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
        self.des_orientation = np.array([data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w])


    def action_manager_callback(self):
        if self.des_pose is not None and self.des_orientation is not None:
            action = list(np.concatenate([self.des_pose, self.des_orientation]) * 100)
            self.ue_env.perform_action(self.action_manager, {self.action_type: action})

    # Costruzione sensori
    def build_sensors(self, sensors_used=None, settings_dir=None, env_topic=None):
        if sensors_used is None:
            sensors_used = self.sensor_used
        if settings_dir is None:
            settings_dir = os.path.join(os.path.dirname(__file__), 'settings')
        if env_topic is None:
            env_topic = self.get_name()

        built = {}

        # Liste di successo per mantenere solo i sensori effettivamente costruiti
        ok_specific_names = [] #nomi specifici tra parentesi
        ok_ue_names = []  #chiavi di ue_env.sensor_set nome+nome tra parentesi

        for obs in sensors_used or []:
           
            if '(' in obs and ')' in obs:
                sensor_type = obs.split('(')[0].strip()
                specific_name = obs.split('(')[1].split(')')[0].strip()
            else:
                sensor_type = obs.strip() #se manca il nome tra graffe itera l'intera stringa
                specific_name = obs.strip()

            ue_sensor = self.ue_env.sensor_set.get(obs) #.get(obs) cerca nel dizionario la voce con la chiave obs (nome completo)
            obs_settings = getattr(ue_sensor, 'settings', {}) if ue_sensor is not None else {} #ottengo le impostazioni ue del sensore

            # Verifica che il modulo/classe del sensore esista; se non implementato, salta
            try:
                # Risolvi il path modulo in modo robusto sia in sviluppo che da pacchetto installato
                base_pkg = __package__  # es: 'ros2_unreal_interface.src'
                if base_pkg:
                    module_name = f"{base_pkg}.sensors.{sensor_type}"
                else:
                    module_name = f"sensors.{sensor_type}"
                module = importlib.import_module(module_name) #sensor_type è il nome della classe del sensore
            except ModuleNotFoundError:
                self.get_logger().warning(f"Modulo per sensore '{sensor_type}' non trovato. Salto '{obs}'.")
                continue

            try:
                class_ = getattr(module, sensor_type)
            except AttributeError:
                self.get_logger().warning(f"Classe sensore '{sensor_type}' non implementata nel modulo. Salto '{obs}'.")
                continue
   

            json_path = os.path.join(settings_dir, f'{sensor_type}.json') #costruisco il percorso del file json di configurazione

            settings = {} #dizione in cui inserisco i file json di configurazione
            try:
                with open(json_path, 'r') as f:
                    settings = json.load(f) or {}
            except FileNotFoundError: #se non ho implementato ancora la classe in teoria non ho ancora il json
                self.get_logger().warning(f"File di configurazione '{json_path}' non trovato per '{sensor_type}'. Uso impostazioni vuote.")
        
          

            json_path_unreal = os.path.join(settings_dir, 'unreal_settings.json')

            if os.path.exists(json_path_unreal):
                with open(json_path_unreal, 'r') as f_unreal:
                    settings['unreal_settings'] = json.load(f_unreal) or {} #+ evito errori se file json non è scritto

            settings['env_topic'] = env_topic #passa al sesnore il topic dell'ambiente (nome del nodo)
            settings['obs_settings'] = obs_settings #impostazioni lette dal sesnore secondo la conf corrente ROS 
            settings['node'] = self #riferimento al nodo ros
            settings['environment_ue'] = self.ue_env #handle all'ambiente UE complessivo
            settings['ue_sensor'] = ue_sensor #handle diretto al sensore specifico, serve per leggere le proprietà specifiche che ottengo tramite ue
            settings['sensor_type'] = sensor_type
            settings['specific_name'] = specific_name
            
            sanitized_specific_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in specific_name).lower() #converte tutto il minuscolo, permette di evitare spazi e simoli strani
            # Frame defaults per sensore: limita 'camera_frame' alle camere; per GPS usa 'base_frame_default'
            if sensor_type in ('RGBCamera', 'DepthCamera', 'SegmentationCamera'):
                settings['camera_frame'] = f"camera_{sanitized_specific_name}"
            elif sensor_type == 'GPS':
                settings['base_frame_default'] = f"base_{sanitized_specific_name}"
            else:
                # Nessun frame predefinito per altri sensori se non necessario
                pass

      

            # Istanziazione del sensore; se fallisce, salta
            try:
                instance = class_(**settings)
            except Exception as e:
                self.get_logger().warning(f"Impossibile istanziare sensore '{sensor_type}' ('{specific_name}'): {e}. Salto.")
                continue
     

            built[specific_name] = instance
            ok_specific_names.append(specific_name)
            ok_ue_names.append(obs)
            self.get_logger().info(f"Creato sensore {sensor_type} -> '{specific_name}'")
           
        with self._sensor_lock:
            self.sensors = built #dizionario “nome specifico → istanza del sensore”
        # Aggiorno le liste usate nel loop per includere solo i sensori effettivamente costruiti
        
        self._specific_names = ok_specific_names
        self.sensor_used = ok_ue_names
        self.get_logger().info(f"Sensori costruiti: {list(self.sensors.keys())}")
        return built #restituisco il dict di tutti i sesnori costuiti

 

    
    def publish_loop(self):
        #setup del timing
        period = 1.0 / self.fps if self.fps and self.fps > 0 else 0.0
        next_time = time.perf_counter()

        while not self._sensor_thread_stop.is_set(): #il loop contiuna finché non blocco il thread
            # Se non ci sono sensori, evita chiamate a UE e fai solo timing
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
                observations = self.ue_env.get_obs(self.sensor_used) #acquisisco osservazioni da UE
                # se arrivo qui, la connessione è OK -> reset contatore errori e log di ripristino una volta
                if self._ue_failures > 0:
                    self.get_logger().info("Connessione UE ripristinata. Riprendo la pubblicazione.")
                self._ue_failures = 0
            except Exception as e: #gestione errore lato ue
                self._ue_failures += 1
                if self._ue_failures == 1:
                    self.get_logger().error(f"Errore UE get_obs: {e}. Connessione persa? Ritento con backoff.")
                else:
                    # logga meno spesso per evitare spam
                    if self._ue_failures % 10 == 0:
                        self.get_logger().warning(f"UE disconnesso da {self._ue_failures} tentativi: {e}")

                # backoff esponenziale con tetto massimo
                backoff = min(1.0 * (2 ** (self._ue_failures - 1)), 10.0)
                # attende backoff secondi (interrompibile)
                self._sensor_thread_stop.wait(timeout=backoff)
                # salta il resto del ciclo (niente pubblicazione/timing) e riprova
                continue

            # Normalizzazione: liste/tuple o singolo valore -> dict {ue_name: dato}
            if isinstance(observations, (list, tuple)):
                if len(observations) != len(self.sensor_used):
                    self.get_logger().warning(
                        f"Dimensione osservazioni ({len(observations)}) diversa da sensor_used ({len(self.sensor_used)}). Uso zip parziale.")
                observations = {ue: val for ue, val in zip(self.sensor_used, observations)}
            elif observations is not None and not isinstance(observations, dict) and len(self.sensor_used) == 1:
                observations = {self.sensor_used[0]: observations}

            # Se non ci sono osservazioni, salta la pubblicazione in questo tick (tipo obseravtion è None)
            if not observations:
                pass
            else:
                # Pubblicazione
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
                                            self.get_logger().error(f"Errore pubblicando {specific_name}: {e}")
                                    continue
                                if getattr(sensor, '_expects_data', False):
                                    try:
                                        sensor.publish_observation(data)
                                    except Exception as e:
                                        self.get_logger().error(f"Errore pubblicando {specific_name}: {e}")
                                else:
                                    try:
                                        sensor.publish_observation()
                                    except Exception as e:
                                        self.get_logger().error(f"Errore pubblicando {specific_name}: {e}")
                except Exception as e:
                    self.get_logger().error(f"Errore durante la pubblicazione: {e}")

            # timing
            if period > 0:
                next_time += period
                sleep_time = max(0.0, next_time - time.perf_counter())
                if sleep_time > 0:
                    # Usa l'Event per rendere lo stop reattivo anche durante l'attesa
                    self._sensor_thread_stop.wait(timeout=sleep_time)
            else:
                # Attesa breve ma interrompibile
                self._sensor_thread_stop.wait(timeout=0.01)

def main(args=None) -> None:
    rclpy.init(args=args)
    node = Environment()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    
    try:
        if hasattr(node, '_sensor_thread') and node._sensor_thread is not None:
            node._sensor_thread.join(timeout=2.0)
    except Exception:
        pass

if __name__ == '__main__':
    main()