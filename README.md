# Sign-Speech Two-Way Visual Communication System

This repository collects two existing research prototypes that together enable a vision-only
conversation loop between Deaf or Hard-of-Hearing signers and hearing speakers. Phase 1 focuses on
understanding and stabilising each project before they are fused into a single user experience.

---

## Project 1 – Real-Time Sign Language Translator

**Goal:** Convert American Sign Language (ASL) gestures captured from a webcam into spoken English in
real time.

### How it works

1. **Live video capture** – A Streamlit interface opens the user’s webcam and streams frames to the
   recognition pipeline.
2. **Hand and pose landmark extraction** – MediaPipe isolates key points for the hands and upper
   body, producing a set of coordinates that are invariant to camera placement.
3. **Sequence classification** – A TensorFlow/Keras Long Short-Term Memory (LSTM) network (stored in
   `sign.h5`) observes the landmark sequences and predicts a sign label from the trained vocabulary.
4. **Text and speech output** – The predicted English phrase is rendered in the Streamlit UI and can
   be spoken aloud through a text-to-speech (TTS) engine.

### Key components

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit application that manages webcam capture, inference loop, and UI controls. |
| `sign.h5` | Pre-trained sign classification model loaded by the app. |
| `requirements.txt` | Python dependencies (TensorFlow/Keras, OpenCV, MediaPipe, Streamlit, etc.). |
| `Sign Detection.ipynb`, `PumpkinSeeds_Data_Extractor.ipynb` | Notebooks that document dataset preparation and model training experiments. |

### Capabilities & limitations

* **Vocabulary size** – Recognises a fixed set of ASL gestures present in the training data.
* **Latency** – Designed for near real-time feedback, but performance depends on hardware and camera
  quality.
* **Environment expectations** – Requires a stable webcam feed and consistent lighting so that
  MediaPipe landmarks can be detected reliably.

---

## Project 2 – Lip Reading (Leap Reading AI)

**Goal:** Infer spoken English phrases by analysing only the visual motion of a speaker’s mouth.

### How it works

1. **Data ingestion** – Video clips and alignment transcripts are loaded from dataset folders (for
   example `data/s1/`).
2. **Frame preprocessing** – Each clip is converted into a uniform sequence of cropped mouth-region
   frames suitable for model input.
3. **Sequence modelling** – A LipNet-inspired architecture built with TensorFlow processes the frame
   sequence using 3D convolutions and recurrent layers. The model is trained with Connectionist
   Temporal Classification (CTC) loss so it can align predictions with variable-length inputs.
4. **Decoding** – During inference, the raw character probabilities are decoded into readable text
   using TensorFlow’s CTC decoding utilities.

### Key components

| File | Purpose |
| --- | --- |
| `main.py` | End-to-end training script that assembles the dataset pipeline, defines the network, and performs training/evaluation. |
| `StreamlitApp/streamlit.py` | Modern Streamlit demo that lets a user upload/select clips, runs inference, and visualises results. |
| `app.py` + `modelutil.py` | Legacy Streamlit interface that can perform quick predictions with pre-saved weights. |
| `models/` | Stores trained checkpoints that the Streamlit apps can load. |
| `README.md` | In-depth documentation of features, setup, and architecture. |

### Capabilities & limitations

* **LipNet foundation** – Matches the research architecture by combining spatiotemporal convolutions
  with bidirectional GRU layers and CTC decoding.
* **Data requirements** – Needs aligned video/transcript pairs; sample data is referenced in
  `data/s1/`, but full datasets are typically too large for the repository.
* **Usage modes** – Supports both training from scratch (via `main.py`) and inference through
  Streamlit demos for rapid experimentation.

---

## Next steps

During Phase 1 the priority is to stabilise these projects as standalone Python packages. The next
milestones include:

1. Normalise packaging so each project exposes a clean inference API (e.g., `predict_asl()` and
   `predict_lip()`).
2. Assemble a unified environment specification covering all shared dependencies.
3. Create a `fusion_core` module and Streamlit UI skeleton that orchestrates turn-taking between the
   sign translator and lip-reading decoder.

These improvements will set the foundation for a full two-way visual communication loop in later
phases.
