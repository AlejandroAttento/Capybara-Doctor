import os
import json
import shutil
import hashlib
from langchain.schema import AIMessage, HumanMessage


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

def serialize_messages(messages):
    serialized = []
    for msg in messages:
        serialized.append({"type": msg.type, "content": msg.content})
    return serialized

def deserialize_messages(serialized_messages):
    messages = []
    for msg in serialized_messages:
        if msg['type'] == 'human':
            messages.append(HumanMessage(content=msg['content']))
        elif msg['type'] == 'ai':
            messages.append(AIMessage(content=msg['content']))
    return messages

def remove_all_in_directory(directory_path):
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)  # Remove file or symbolic link
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  # Remove directory and all its contents
        except Exception as e:
            print(f"Failed to delete {item_path}. Reason: {e}")

    # Wait until the directory is empty
    while os.listdir(directory_path):
        time.sleep(0.1)  # Sleep briefly to avoid busy waiting
    print(f"All contents in '{directory_path}' have been deleted.")