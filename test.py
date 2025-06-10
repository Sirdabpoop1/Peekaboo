import tensorflow as tf
import os

def is_valid_jpeg(path):
    try:
        img_bytes = tf.io.read_file(path)
        _ = tf.io.decode_jpeg(img_bytes)
        return True
    except:
        return False

deleted = []

for root, dirs, files in os.walk("aug_data"):  # or replace with 'neg_data' if needed
    for file in files:
        if file.lower().endswith(".jpg"):
            path = os.path.join(root, file)
            if not is_valid_jpeg(path):
                os.remove(path)
                deleted.append(path)

print("Deleted invalid JPEGs:")
for f in deleted:
    print(f)

print(f"Total deleted: {len(deleted)}")
