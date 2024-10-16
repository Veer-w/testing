import random
from datetime import datetime

class SyntheticDataGenerator:
    def __init__(self, sensor_names):
        self.sensor_names = sensor_names

    def generate_data(self):
        sensor_data = {sensor: random.uniform(0, 100) for sensor in self.sensor_names}
        sensor_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return sensor_data

if __name__ == "__main__":
    sensor_names = ["Temperature Sensor", "Pressure Sensor", "Vibration Sensor", "Humidity Sensor"]
    generator = SyntheticDataGenerator(sensor_names)
    for _ in range(10):
        print(generator.generate_data())
