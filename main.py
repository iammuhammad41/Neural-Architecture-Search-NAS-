"""
Neural Architecture Search on CIFAR-10 (Kaggle) with Keras Tuner Hyperband

Steps:
1. Extract CIFAR‑10 .7z archives in Kaggle.
2. Read trainLabels.csv and reorganize train images into class‑subfolders.
3. Build tf.data pipelines via image_dataset_from_directory.
4. Define a tunable CNN hypermodel.
5. Search for the best architecture with Hyperband.
6. Evaluate on the held‑out test set.
"""

import os
import shutil
import pandas as pd
import py7zr
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

# ─── EXTRACT ARCHIVES ────────────────────────────────────────────────────────────
os.makedirs("/kaggle/working/train_raw", exist_ok=True)
os.makedirs("/kaggle/working/test", exist_ok=True)

with py7zr.SevenZipFile("/kaggle/input/cifar-10/train.7z", mode='r') as z:
    z.extractall(path="/kaggle/working/train_raw")

with py7zr.SevenZipFile("/kaggle/input/cifar-10/test.7z", mode='r') as z:
    z.extractall(path="/kaggle/working/test")

# ─── READ LABELS & REORGANIZE ─────────────────────────────────────────────────────
labels_df = pd.read_csv("/kaggle/input/cifar-10/trainLabels.csv")  # columns: id,label

# Create a directory for each class
sorted_train_dir = "/kaggle/working/train"
for cls in labels_df['label'].unique():
    os.makedirs(os.path.join(sorted_train_dir, str(cls)), exist_ok=True)

# Move each image into its class folder
raw_train_folder = "/kaggle/working/train_raw/train"
for _, row in labels_df.iterrows():
    src = os.path.join(raw_train_folder, f"{row['id']}.png")
    dst = os.path.join(sorted_train_dir, str(row['label']), f"{row['id']}.png")
    shutil.copy(src, dst)

# ─── CREATE TF.DATASETS ───────────────────────────────────────────────────────────
IMG_SIZE = (32, 32)
BATCH_SIZE = 64

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    sorted_train_dir,
    labels="inferred",
    label_mode="int",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_split=0.1,
    subset="training",
    seed=42,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    sorted_train_dir,
    labels="inferred",
    label_mode="int",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_split=0.1,
    subset="validation",
    seed=42,
)

# OPTIONAL: use the provided test set if it has labels;
# otherwise just evaluate on the validation set.

# ─── DEFINE HYPERMODEL ────────────────────────────────────────────────────────────
def build_model(hp):
    inputs = keras.layers.Input(shape=(*IMG_SIZE, 3))
    x = inputs

    # Tune number of convolutional blocks
    for i in range(hp.Int("conv_blocks", 1, 3)):
        filters = hp.Choice(f"filters_{i}", [32, 64, 128])
        x = keras.layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
        x = keras.layers.MaxPooling2D()(x)

    x = keras.layers.Flatten()(x)

    # Tune number of dense layers
    for j in range(hp.Int("dense_layers", 1, 2)):
        units = hp.Int(f"dense_units_{j}", 64, 256, step=64)
        x = keras.layers.Dense(units, activation="relu")(x)
        x = keras.layers.Dropout(hp.Float(f"dropout_{j}", 0.0, 0.5, step=0.1))(x)

    outputs = keras.layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    lr = hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# ─── SET UP & RUN HYPERBAND SEARCH ────────────────────────────────────────────────
project_dir = "/kaggle/working/nas_cifar10"
tuner = kt.Hyperband(
    build_model,
    objective="val_accuracy",
    max_epochs=20,
    factor=3,
    directory=project_dir,
    project_name="hyperband",
)

early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)

tuner.search(
    train_ds,
    epochs=20,
    validation_data=val_ds,
    callbacks=[early_stop],
    verbose=2,
)

# ─── RETRIEVE & EVALUATE BEST MODEL ───────────────────────────────────────────────
best_model = tuner.get_best_models(num_models=1)[0]
best_hps = tuner.get_best_hyperparameters(1)[0]
print("Best hyperparameters:")
print(f"  conv_blocks: {best_hps.get('conv_blocks')}")
print(f"  filters_0: {best_hps.get('filters_0')}")
print(f"  dense_layers: {best_hps.get('dense_layers')}")
print(f"  dense_units_0: {best_hps.get('dense_units_0')}")
print(f"  learning_rate: {best_hps.get('learning_rate')}")

# Evaluate on validation set
val_loss, val_acc = best_model.evaluate(val_ds, verbose=2)
print(f"\nValidation Accuracy: {val_acc:.4f}")

# Save the best model
best_model.save(os.path.join(project_dir, "best_nas_model.h5"))
