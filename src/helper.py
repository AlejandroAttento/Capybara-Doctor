import json
import hashlib


class ConfigManager:
    def __init__(self, json_config_path: str):
        self.json_config_path = json_config_path
        self.json_config = None

        self._load_config_json()

    def _load_config_json(self):
        try:
            with open(self.json_config_path, "r") as file_obj:
                self.json_config = json.load(file_obj)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.json_config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON file: {e}")

    def get_config(self, keys: str | list, default=None):
        if isinstance(keys, str):
            keys = [keys]

        data = self.json_config
        for key in keys:
            if isinstance(data, dict):
                data = data.get(key, default)
            else:
                return default
        return data

def generate_id(input):
    return hashlib.md5(input.encode()).hexdigest()