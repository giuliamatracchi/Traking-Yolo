from abc import ABCMeta, abstractmethod

class Sensor(metaclass=ABCMeta):
  
    sensor_name: str
    topic: str

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def change_settings(self):
        pass

    @abstractmethod
    def publish_observation(self):
        pass