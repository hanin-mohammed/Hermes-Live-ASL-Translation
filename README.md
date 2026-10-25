<p align="center">
  <img src="assets/icons/macOS%20icons/icon_256x256.png" width="128" alt="HERMES application icon">
</p>

<h1 align="center">Project H.E.R.M.E.S.</h1>

<p align="center"><strong>Hand-sign Extraction and Recognition using Machine-learning for Enhanced Speech</strong></p>

HERMES is a Python desktop application that recognises seven selected static hand signs from a live webcam feed. It combines MediaPipe hand landmarks with a K-nearest neighbours classifier, retrieves contextual information from MySQL and speaks accepted predictions aloud.

I developed HERMES as my CBSE Class XII Computer Science project to study how computer vision, a simple supervised learning method and a relational database can operate within one understandable application.

## What it does

- Captures and mirrors live webcam frames with OpenCV
- Locates 21 hand landmarks with MediaPipe
- Converts each detected hand into 63 numerical coordinates
- Classifies `A`, `B`, `C`, `1`, `2`, `3` and `ILY` using a three-neighbour KNN model
- Rejects distant matches with a nearest-distance threshold
- Retrieves descriptions and speech phrases from MySQL
- Displays the result in a Tkinter interface
- Speaks a stable new prediction without blocking the camera feed on macOS

## Recognition pipeline

```text
Webcam frame
    -> MediaPipe hand detection
    -> 21 landmarks x (x, y, z)
    -> 63-feature vector
    -> KNN classification
    -> distance-based acceptance check
    -> MySQL information lookup
    -> Tkinter display and speech output
```

## Dataset and model

The included dataset contains **2,944 labelled examples**. Each row holds 63 landmark coordinates followed by its class label.

| Label | Examples |
|---|---:|
| A | 422 |
| B | 420 |
| C | 421 |
| 1 | 420 |
| 2 | 421 |
| 3 | 420 |
| ILY | 420 |

The model uses `KNeighborsClassifier(n_neighbors=3)`. HERMES checks the closest training example and treats distances above `0.35` as uncertain. This reduces confident predictions for unfamiliar or poorly positioned gestures.

## Technology

- Python 3.11
- OpenCV for camera input and frame processing
- MediaPipe Hand Landmarker for landmark extraction
- scikit-learn for KNN classification
- MySQL for sign descriptions and speech text
- Tkinter and Pillow for the desktop interface

## Run locally

Clone the repository, then follow [HOW_TO_RUN.md](HOW_TO_RUN.md) to install dependencies, create the local MySQL database and start the application.

The trained KNN model, MediaPipe task file and dataset are included, so training is not required before the first run.

## Project structure

```text
HERMES/
|-- hermes_app.py                  Main application
|-- requirements.txt              Python dependencies
|-- assets/                       App icon and interface fonts
|-- data/landmark_data.csv        Labelled landmark dataset
|-- database/setup_hermes_database.sql
|-- models/hand_landmarker.task   MediaPipe hand model
|-- models/landmark_knn_model.pkl Trained KNN classifier
`-- scripts/
    |-- collect_landmark_data.py
    `-- train_landmark_model.py
```

## Current scope

HERMES recognises seven static signs under suitable lighting and camera positioning. It is a focused educational prototype rather than a complete sign-language translator. Performance can vary with hand orientation, distance, background and lighting. Future development could add a larger and more varied dataset, sequence recognition for dynamic signs, systematic evaluation on unseen users and platform-independent speech output.

## Author

Created by **Hanin Mohammed**.
