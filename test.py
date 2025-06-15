import os
import json

label_dir = 'data/val/labels'

for filename in os.listdir(label_dir):
    if filename.endswith('.json'):
        path = os.path.join(label_dir, filename)
        with open(path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if 'class' not in data:
                    print(f"Missing 'class' in {filename}")
                if 'bbox' not in data:
                    print(f"Missing 'bbox' in {filename}")
            except json.JSONDecodeError:
                print(f"Invalid JSON in {filename}")
