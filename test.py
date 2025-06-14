import os
import json

# Path to your image and label folders
image_dir = 'data/test/images'
label_dir = 'data/test/labels'

# Loop through all image files
for img_file in os.listdir(image_dir):
    if img_file.lower().endswith('.jpg'):
        base_name = os.path.splitext(img_file)[0]
        label_path = os.path.join(label_dir, base_name + '.json')

        # If no label file exists, create a dummy one
        if not os.path.exists(label_path):
            dummy_label = {
                "class": 0,
                "bbox": [0, 0, 0, 0]
            }
            with open(label_path, 'w', encoding='utf-8') as f:
                json.dump(dummy_label, f, indent=4)
            print(f"Created dummy label for: {img_file}")
