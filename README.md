﻿# TORCS Racing Agent with Machine Learning

This project integrates manual keyboard-based driving with a machine learning (ML)-based control system in the TORCS racing simulator. It enables human driving with keyboard inputs while collecting sensor and control data, then uses that data to train a PyTorch neural network. Once trained, the model can control the vehicle autonomously, mimicking human driving behavior.

---

## Contributors

- Ayaan Khan (22i-0832)
- Ayaan Mughal (22i-0861)
- Malaika Afzal (22i-0885)

---

## Overview

* **Manual & ML Control**: Drive the vehicle using keyboard inputs or let the trained model take over.
* **Data Logging**: Collects sensor and control data in CSV format during manual driving.
* **Supervised Learning Model**: Trains a PyTorch model to predict steering, acceleration, and braking based on past driving data.
* **Real-Time Inference**: Switch seamlessly between manual and ML-based driving during a session.

---

## Project Components

* `train_torch.py`: Trains the PyTorch model on CSV driving data.
* `models.py`: Defines the ML model architecture.
* `driver.py`: Handles vehicle control, integrating both keyboard and ML control.
* `utils.py`: Provides preprocessing and scaling utilities.
* `pyclient.py`: Connects the client to the TORCS server.

---

## Prerequisites

* Python 3.8+
* PyTorch (`torch`)
* NumPy (`numpy`)
* Pandas (`pandas`)
* scikit-learn (`scikit-learn`)
* keyboard (`keyboard`)
* xml.etree.ElementTree (standard)
* TORCS Simulator (installed and configured)
* (Optional) CUDA (if you want to train the model on GPU)

---

## Tools and Frameworks Used

* **PyTorch**: For building and training the ML model.
* **scikit-learn**: For data preprocessing.
* **NumPy/Pandas**: For data manipulation.
* **keyboard**: For capturing keyboard inputs.
* **TORCS**: The racing simulator environment.
* **XML**: For reading configuration files (track, car details).

---

## Setup and Usage

### Data Collection

Run the TORCS server and connect with the Python client:

```bash
python pyclient.py --port 3001 --stage 0 --maxEpisodes 1 --maxSteps 0
```

* Control the vehicle manually with the keyboard.
* Sensor and control data is logged into CSV files (one per track and car combination).

---

### Model Training

Run:

```bash
python train_torch.py
```

* This script:

  * Loads all CSV files in the current directory.
  * Preprocesses the data, handling missing values and scaling.
  * Builds sequences of past frames (length 5) to predict the next control action.
  * Trains a PyTorch LSTM model with both numeric and categorical features (track and car names).
  * Applies a penalty for off-center track position and angle extremes to encourage better driving behavior.
  * Saves:

    * `torcs_model.pt`: the trained model weights.
    * `preproc.pkl`: the preprocessing stats (scaler and category maps).

---

### Enabling ML Control

In `driver.py`, ensure the following:

```python
USE_ML = True
```

This flag enables the ML model to control the vehicle during the simulation (provided the model file `torcs_model.pt` and preprocessor file `preproc.pkl` are available).

---

### Running the Simulation with ML Control

1. Start the TORCS server.
2. Run the Python client:

```bash
python pyclient.py --port 3001 --stage 0 --maxEpisodes 1 --maxSteps 0
```

* The model will load and take control of the vehicle automatically unless manual keyboard inputs are detected.

---

## Detailed Component Explanations

### `train_torch.py`

* **Data Loading**: Reads all `.csv` files, expands list columns, fills missing values, and encodes categorical data.
* **Preprocessing**: Uses Min-Max scaling for numeric features and one-hot or embedding for categorical ones.
* **Sequence Construction**: Builds sequences of past frames (length 5) to capture temporal patterns in driving.
* **Model Architecture**:

  * Embeddings for track and car names.
  * LSTM layers to capture temporal patterns.
  * MLP head to predict steering, acceleration, and braking.
* **Loss**:

  * Mean Squared Error (MSE) loss.
  * Penalty for off-center track position and angle extremes.
* **Training**:

  * Trains over 20 epochs with batch size 64.
  * Prints training and validation losses each epoch.
* **Artifacts**:

  * `torcs_model.pt` (model weights)
  * `preproc.pkl` (scaler and category mappings)

---

### `models.py`

Defines `TORCSModel`:

* Embeds categorical features (track and car).
* Concatenates embeddings with numeric features.
* Uses LSTM layers to capture temporal dependencies.
* Fully connected MLP head to output steering, acceleration, and brake.
* Applies appropriate activations:

  * Tanh for steering (\[-1,1])
  * Sigmoid for acceleration and brake (\[0,1])

---

### `driver.py`

* Contains the driving logic:

  * Keyboard-based control.
  * ML-based control if enabled.
* Collects sensor data and logs it to CSV files.
* Reads the latest model weights and preprocessing stats.
* Supports automatic reset if the vehicle gets stuck.
* Handles gear control, clutch, and logging.

---

## Dataset Format

Each row in the CSV contains:

* **Sensor values**: speed, angle, track sensors, focus sensors, etc.
* **Control outputs**: steering, acceleration, braking.
* **Meta-data**: clutch, gear, reset state, focus direction, keypresses.

---

## Notes

* Make sure TORCS is installed and configured properly on your system.
* The TORCS server should be running before you launch the Python client.
* If training on GPU, ensure CUDA is installed and PyTorch is installed with CUDA support.
* Some features, like automatic resets, can be toggled in `driver.py`.

---

## Conclusion

This project demonstrates a hybrid manual-AI driving model in the TORCS simulator using supervised learning and PyTorch. 

---

