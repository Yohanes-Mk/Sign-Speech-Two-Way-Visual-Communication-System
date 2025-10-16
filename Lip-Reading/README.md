# Lip Reading

Lip Reading is a deep learning project based on the LipNet architecture for visual speech recognition (lip reading). It processes video frames of a speaker's mouth and predicts the spoken text using a neural network trained with Connectionist Temporal Classification (CTC) loss. The project includes a Streamlit web application for interactive video upload, processing, and real-time predictions.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Running the Streamlit App](#running-the-streamlit-app)
- [Core Components](#core-components)
- [Model Details](#model-details)
- [Data Pipeline](#data-pipeline)
- [Acknowledgements](#acknowledgements)

---

## Features

- **Lip Reading Model**: End-to-end deep learning model for visual speech recognition.
- **Streamlit Web App**: Upload and process videos, visualize model predictions, and see intermediate outputs.
- **Data Pipeline**: Automated loading, preprocessing, and batching of video and alignment data.
- **Model Checkpointing**: Save and load model weights for reproducibility.
- **Visualization**: Display video frames and animated GIFs of mouth movements.

---

## Project Structure

```
LipReading-2.0/
│
├── data/                  # Video and alignment data (not included in repo)
│   └── s1/                # Example speaker folder with .mpg files
│
├── models/                # Saved model checkpoints
│
├── StreamlitApp/
│   ├── streamlit.py       # Streamlit web app (main entry point)
│   ├── utils.py           # Data loading and utility functions
│   └── model.py           # Model loading function for Streamlit
│
├── main.py                # Main training, evaluation, and prediction script
├── app.py                 # Alternate Streamlit app (legacy/alternative)
├── modelutil.py           # Model loading utility for app.py
├── .gitignore
└── README.md
```

---

## Setup & Installation

1. **Clone the Repository**
   ```bash
   git clone <the repo link>
   cd lip-reading-AI
   ```

2. **Install Dependencies**
   - Python 3.8+
   - [TensorFlow](https://www.tensorflow.org/)
   - [Streamlit](https://streamlit.io/)
   - OpenCV, imageio, matplotlib, Pillow

   Install with pip:
   ```bash
   pip install -r requirements.txt
   ```

   *(If `requirements.txt` is missing, install manually:)*

   ```bash
   pip install tensorflow streamlit opencv-python imageio matplotlib pillow
   ```

3. **Prepare Data**
   - Place your video files (e.g., `.mpg`) and alignment files in the `data/s1/` directory.
   - The model expects videos and corresponding alignment files.

---

## Usage

### Training the Model

1. **Edit `main.py`** to ensure paths and parameters match your data.
2. **Train the model:**
   ```bash
   python main.py
   ```
   - Model checkpoints will be saved in the `models/` directory.

### Running the Streamlit App

1. **Navigate to the StreamlitApp directory:**
   ```bash
   cd StreamlitApp
   ```

2. **Start the app:**
   ```bash
   streamlit run streamlit.py
   ```

3. **Using the App:**
   - Upload or select a video from the sidebar.
   - The app will display the video, an animation of the mouth region, and the model's predicted text.

---

## Core Components

### main.py

- **Data Pipeline**: Loads and preprocesses video and alignment data, shuffles, batches, and splits into train/test sets.
- **Model Definition**: Defines the neural network architecture (LipNet-like).
- **Training Loop**: Compiles the model with Adam optimizer and CTC loss, sets up callbacks for checkpointing and learning rate scheduling.
- **Prediction**: Loads model weights and runs predictions on test data.

### StreamlitApp/streamlit.py

- **User Interface**: Lets users select/upload videos and displays results.
- **Video Processing**: Converts videos to mp4, extracts frames, and creates GIFs.
- **Prediction**: Loads the trained model and decodes predictions using CTC decoding.

### StreamlitApp/utils.py

- **load_data**: Loads video frames and alignments.
- **num_to_char / char_to_num**: Converts between character tokens and indices for encoding/decoding.

---

## Model Details

- **Input**: Sequences of video frames (mouth region).
- **Architecture**: Based on LipNet, uses 3D convolutions, recurrent layers, and dense layers.
- **Loss Function**: Connectionist Temporal Classification (CTC) for sequence-to-sequence alignment.
- **Decoding**: Uses TensorFlow's `ctc_decode` for converting model outputs to text.

---

## Data Pipeline

- **Video Loading**: Uses OpenCV to read and preprocess video frames.
- **Alignment Loading**: Reads alignment files for ground truth text.
- **Batching**: Pads sequences to uniform length for batching.
- **Mapping**: Converts text to numerical tokens and vice versa.

---

## Acknowledgements

- Based on the [LipNet](https://arxiv.org/abs/1611.01599) architecture.
- Inspired by open-source implementations and tutorials.

---

## Notes

- `.gitignore` excludes large files, checkpoints, and video files from version control.
- For best results, use high-quality, well-aligned video and text data.
- The app and model are for research/educational use and not production-ready.

