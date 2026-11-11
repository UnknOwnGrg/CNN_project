import os
import json
import base64
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.callbacks import Callback
from PIL import Image
import io
import matplotlib.cm as cm


# Constants
IMG_SIZE = (128, 128)
CLASSES= ['messi', 'yamal','lewandowski']
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "player_cnn.keras")
STATUS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_status.json")
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..","dataset")

# In-memory model cache to avoid re-loading from disk on every inference request
_model_cache = None

def get_model():
    """Retrieves the trained model from in-memory cache, or loads it from disk if available."""
    global _model_cache
    if _model_cache is None:
        if os.path.exists(MODEL_PATH):
            print(f"Loading trained model from disk: {MODEL_PATH}")
            try :
                _model_cache = load_model(MODEL_PATH)
            except Exception as e:
                print(f"Error loading model from disk: {e}")
        else:
            print("No saved model found on disk.")
    return _model_cache

class RealTimeStatsCallback(Callback):
    """Callback to write training statistics to a JSON file at the end of each epoch."""
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        # Clear stats at start
        with open(STATUS_PATH, 'w') as f:
            json.dump({
                "status": "training",
                "current_epoch": 0,
                "total_epoch": total_epochs,
                "history": {
                    "loss": [], "accuracy": [],
                    "val_loss": [], "val_accuracy": []
                }
            }, f)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        try:
            # Read Existing History
            if os.path.exists(STATUS_PATH):
                with open(STATUS_PATH, 'r') as f:
                    data = json.load(f)
            else:
                data = {"history":
                            {"loss": [],
                             "accuracy": [],
                             "val_loss": [],
                             "val_accuracy": []
                             }
                        }

            data['status'] = 'training'
            data['current_epoch'] = epoch + 1
            data['total_epochs'] = self.total_epochs

            # Append new metrics
            data['history']['loss'].append(float(logs.get('loss', 0.0)))
            data['history']['accuracy'].append(float(logs.get('accuracy', 0.0)))
            data['history']['val_loss'].append(float(logs.get('val_loss', 0.0)))
            data['history']['val_accuracy'].append(float(logs.get('val_accuracy', 0.0)))

            with open(STATUS_PATH, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Error writing training status: {e}")


# Model Creation by using open source model MobileNetV2
def create_model():
    """Builds a MobileNetV2-based transfer learning model for Barcelona footballers classification."""
    # Loading a pre-trained MobileNetV2 base (excluding classification head)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(128, 128, 3),
        include_top=False,
        weights='imagenet'
    )

    # Freeze pre-trained layers
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(name='global_pooling'),
        Dense(64, activation='relu', name='dense_1'),
        Dropout(0.5, name='dropout'),
        Dense(len(CLASSES), activation='softmax',name='output')
    ])


    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# Simply to load the dataset which had been passed by the User's
def load_dataset_data():
    """Loads images from dataset directory, pre-processes them for MobileNetV2"""

    images = []
    labels = []

    for class_idx, class_name in enumerate(CLASSES):
        class_folder = os.path.join(DATASET_DIR, class_name)
        if not os.path.exists(class_folder):
            continue

        for filename in os.listdir(class_folder):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            filepath = os.path.join(class_folder, filename)
            try:
                #     Load image , resize and convert to numpy array
                img = load_img(filepath, target_size=IMG_SIZE)
                img_arr = img_to_array(img)
                # Preprocess for MobileNetV2 (scales pixels to [-1, 1])
                img_arr = tf.keras.applications.mobilenet_v2.preprocess_input(img_arr)
                images.append(img_arr)


                # Make one-hot encoded label representation
                label = np.zeros(len(CLASSES))
                label[class_idx] = 1.0
                labels.append(label)
            except Exception as e:
                print(f'Skipping corrupted image {filepath}: {e}')

    if len(images) == 0:
        return None, None

    return  np.array(images, dtype='float32'),np.array(labels,dtype='float32')