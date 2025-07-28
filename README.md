# Neural Architecture Search on CIFAR‑10
NAS algorithms search for the most optimal neural network architecture for a given task, potentially outperforming human-designed architectures. This is useful for discovering novel architectures that are more efficient or accurate than manually designed ones.
This project demonstrates how to use **Keras Tuner’s Hyperband** algorithm to automatically discover an optimal CNN architecture for image classification on the CIFAR‑10 dataset.



## 📁 Repository Structure

```
.
├── train_nas_kaggle.py    # Main script: data prep, NAS search, evaluation
├── best_nas_model.h5      # Saved best model (after tuning)
└── README.md              # This file
```



## 🔍 Overview

Neural Architecture Search (NAS) automates the design of neural networks. Here we:

1. **Extract** CIFAR‑10 `.7z` archives in Kaggle.
2. **Reorganize** training images into per‑class folders.
3. Build `tf.data.Dataset` pipelines.
4. Define a **tunable** CNN hypermodel.
5. Run **Hyperband** to optimize:

   * Number of convolutional blocks & filters
   * Number of dense layers, units & dropout
   * Learning rate
6. **Evaluate** the best model on a validation split.
7. **Save** the final model for later use.



## 🛠️ Requirements

* Python 3.7+
* TensorFlow 2.x
* Keras Tuner
* pandas
* py7zr

You can install the Python dependencies with:

```bash
pip install tensorflow keras-tuner pandas py7zr
```



## 🚀 Usage

1. **Place** the CIFAR‑10 archives in:

   ```
   /kaggle/input/cifar-10/train.7z
   /kaggle/input/cifar-10/test.7z
   /kaggle/input/cifar-10/trainLabels.csv
   ```

2. **Run** the NAS script:

   ```bash
   python train_nas_kaggle.py
   ```

   This will:

   * Unpack the archives
   * Sort images by class
   * Launch Hyperband search (logs in `./nas_cifar10/`)
   * Print the best hyperparameters and validation accuracy
   * Save the best model to `nas_cifar10/best_nas_model.h5`

3. **Load** your tuned model later:

   ```python
   from tensorflow import keras
   model = keras.models.load_model("nas_cifar10/best_nas_model.h5")
   ```



## 📊 Results

After the Hyperband search completes, you’ll see output like:

```
Best hyperparameters:
  conv_blocks: 2
  filters_0: 64
  dense_layers: 1
  dense_units_0: 128
  learning_rate: 0.001

Validation Accuracy: 0.8325
```
