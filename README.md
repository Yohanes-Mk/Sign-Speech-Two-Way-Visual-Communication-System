# Sign-Speech Two-Way Visual Communication System

Sign-Speech collects two research prototypes that, together, enable a vision-only conversation loop between Deaf or Hard-of-Hearing signers and hearing speakers. We are living in **Phase 1 – Module Stabilisation**, keeping each experience sharp while shaping the bridge that will fuse them into a single dialogue interface.

`Phase timeline: Phase 0 ✅ | Phase 1 🔄 | Phase 2 ⏳ | Phase 3 ⏳ | Phase 4 ⏳`

---

## Project 1 – Real-Time Sign Language Translator

### Goal
Convert American Sign Language (ASL) gestures captured from a webcam into spoken English in real time.

### How it works
- **Live video capture** – A Streamlit interface opens the user’s webcam and streams frames into the recognition pipeline.  
- **Hand and pose landmark extraction** – MediaPipe isolates key points for the hands and upper body, producing a set of coordinates that remain stable across camera setups.  
- **Sequence classification** – A TensorFlow/Keras Long Short-Term Memory (LSTM) network (served from `sign.h5`) observes landmark sequences and predicts a sign label from the trained vocabulary.  
- **Text and speech output** – The predicted English phrase appears inside the Streamlit UI and is voiced through a text-to-speech (TTS) engine.

### Key components

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit application managing the webcam capture loop, inference pipeline, and UI feedback. |
| `sign.h5` | Pre-trained sign classification weights loaded on startup. |
| `requirements.txt` | Dependency list (TensorFlow/Keras, OpenCV, MediaPipe, Streamlit, audio libs). |
| `Sign Detection.ipynb`, `PumpkinSeeds_Data_Extractor.ipynb` | Notebooks covering dataset preparation and training routines. |

### Capabilities & limitations
- **Vocabulary size** – Recognises a fixed set of ASL gestures reflected in the training data.  
- **Latency** – Tuned for near real-time feedback; performance depends on hardware and webcam quality.  
- **Environment expectations** – Prefers consistent lighting and uncluttered backgrounds so MediaPipe can reliably detect landmarks.

---

## Project 2 – Lip Reading (Leap Reading AI)

### Goal
Infer spoken English phrases by analysing only the visual motion of a speaker’s mouth.

### How it works
- **Data ingestion** – Video clips and alignment transcripts load from dataset folders such as `data/s1/`.  
- **Frame preprocessing** – Each clip is converted into a uniform sequence of cropped mouth-region frames ready for inference.  
- **Sequence modelling** – A LipNet-inspired TensorFlow architecture combines 3D convolutions and recurrent layers, training with Connectionist Temporal Classification (CTC) loss to manage variable-length inputs.  
- **Decoding** – During inference, TensorFlow CTC utilities convert raw character probabilities into readable text.

### Key components

| File | Purpose |
| --- | --- |
| `main.py` | End-to-end training script covering dataset assembly, model setup, and evaluation. |
| `StreamlitApp/streamlit.py` | Modern Streamlit demo for uploading/selecting clips, running inference, and previewing results. |
| `app.py`, `modelutil.py` | Legacy Streamlit interface for fast predictions with stored checkpoints. |
| `models/` | Checkpoint directory used by demos and experiments. |
| `README.md` | In-depth documentation on setup, architecture, and usage patterns. |

### Capabilities & limitations
- **LipNet foundation** – Mirrors the research model, blending spatiotemporal convolutions with bidirectional GRUs and CTC decoding.  
- **Data requirements** – Requires aligned video/transcript pairs; sample data lives under `data/s1/`, while full corpora generally stay external due to size.  
- **Usage modes** – Supports both full training runs (`main.py`) and quick inference trials through Streamlit demos.

---

## Next Steps (Phase 1 Focus)

1. Normalise packaging so each project exposes a clean inference API (e.g., `predict_asl()` and `predict_lip()`).  
2. Assemble a unified environment specification covering all shared dependencies.  
3. Create a `fusion_core` module plus a Streamlit UI skeleton that will orchestrate turn-taking between the sign translator and lip-reading decoder.

These upgrades unlock Phase 2, where both modules will feel like a single interpreter—hands to speech, lips to text, eyes on connection.
