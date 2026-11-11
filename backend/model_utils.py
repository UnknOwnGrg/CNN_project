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

def train_cnn_model(epochs=15 , batch_size=4):
    """Load the player dataset and trains the MobileNetV2 transfer model,logging stats in real-time."""
    global _model_cache

    # Write init status
    with open(STATUS_PATH, 'w') as f:
        json.dump({"status": "starting","current_epoch": 0,"total_epochs": epochs,"history": {}}, f)

    X,y = load_dataset_data()
    if X is None or len(X) < 3:
        with open(STATUS_PATH, 'w') as f:
            json.dump({"status": "failed", "error":"Not enough images in dataset. Please add images first"}, f)
        return False

    # Split into train and validation (80-20 split)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X,y = X[indices], y[indices]

    val_split = int(0.2 * len(X))
    if val_split < 1:
        # Ensures at least 1 validation image
        val_split = 1

    X_val, y_val = X[:val_split], y[:val_split]
    X_train, y_train = X[val_split:], y[val_split:]

    # Define and Train Model
    model = create_model()

    # Use real-time status callback to log metrics to JSON
    status_callbcak = RealTimeStatsCallback(epochs)

    print(f"Transfer training started: {len(X_train)} training samples, {len(X_val)} validation samples.")
    model.fit(
        X_train, y_train,
        validation_data = (X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[status_callbcak],
        verbose=1
    )

    # Save the model
    model.save(MODEL_PATH)

    # Update in-memory model cache
    _model_cache = model

    # Update final training status
    if os.path.exists(STATUS_PATH):
        with open(STATUS_PATH, 'r') as f:
            data = json.load(f)
    else:
        data = {"history": {}}

    data["status"] = "completed"
    with open(STATUS_PATH, 'w') as f:
        json.dump(data, f)

    print("Training finished successfully. Model saved and cached.")

    return True

def run_inference(image_bytes):
    """Runs prediction on custom image bytes. Returns list of dictionary predictions"""
    model = get_model()
    if model is None:
        return{
            "error": "Model not trained yet.Please train the transfer model first"
        }

    try:
        # Load and preprocess image
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert('RGB')
        img_resized = img.resize(IMG_SIZE)
        img_arr = img_to_array(img_resized)
        img_arr = tf.keras.applications.mobilenet_v2.preprocess_input(img_arr)
        # Add batch dimension(1, 128, 128, 3)
        img_arr = np.expand_dims(img_arr, axis=0)

        # Predict
        predictions = model.predict(img_arr)[0]

        results = []

        for idx,score in enumerate(predictions):
            results.append({
                'class': CLASSES[idx],
                'confidence': float(score)
            })

        # Sort predictions by confidence desc
        results = sorted(results, key=lambda x:x['confidence'], reverse=True)
    except Exception as e:
        return {"error": f"Error running inference : {str(e)}"}


def extract_activation_maps(image_bytes):
    """Extracts output activations of the first convolutional layer(16 filters) of the base MobileNetV2 Model. """
    model = get_model()
    if model is None:
        return None

    try:
        # Preprocess input image
        img= Image.open(io.BytesIO(image_bytes))
        img= img.convert('RGB')
        img_resized= img.resize(IMG_SIZE)
        img_arr= img_to_array(img_resized)
        img_arr= tf.keras.applications.mobilenet_v2.preprocess_input(img_arr)
        img_arr= np.expand_dims(img_arr, axis=0)

        # Isolate base MobileNetV2 model (layer 0)
        base_model = model.layers[0]

        # Find the first Conv2D layer inside the base model
        first_conv = None
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                first_conv = layer
                break

        if first_conv is None:
            print("Could not find any Conv2D layer in base model.")
            return None

        # Create intermediate activation model
        activation_model = tf.keras.Model(inputs=base_model.input, outputs=first_conv.output)
        activations = activation_model.predict(img_arr)[0]

        # Export each of the first 16 filters as a base64 encoded image
        maps_base64 = []
        for i in range(16):
            feat_map = activations[:,:,i]

            # Normalize to 0-255 range
            f_min , f_max = feat_map.min(), feat_map.max()
            if f_max - f_min > 0:
                feat_map = 255 * (feat_map - f_min) /  (f_max - f_min)
            feat_map = feat_map.astype('uint8')

            # Convert map to colorful image using colormap (Viridis gives a stunning scientific lookup)
            colored_map = cm.viridis(feat_map)
            colored_map = (colored_map[:,:,:3] * 255).astype('uint8')

            # Create PIL image
            pil_map = Image.fromarray(colored_map)
            pil_map_resized = pil_map.resize((128, 128), Image.Resampling.NEAREST)

            # Save to memory as PNG
            buffer = io.BytesIO()
            pil_map_resized.save(buffer,format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            maps_base64.append(img_str)

        return maps_base64
    except Exception as e:
        print(f"Error extracting activation maps: {e}")
        return None

