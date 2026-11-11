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