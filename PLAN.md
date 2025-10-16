# Phase 1 Integration Plan

This plan outlines the recommended actions to deliver the Phase 1 integration objectives for the Leap Sign-Speech visual communication system. The steps are prioritized to unblock later phases while keeping each legacy module functional.

## 1. Stabilize Existing Demos

1. **Sign translator path handling**
   - Resolve model path loading in `Real-Time-Sign-Language-Translation-master/app.py` using `Path(__file__)` to keep imports portable.
   - Add a helper that locates model assets relative to the module.
2. **Streamlit capture lifecycle**
   - Introduce `st.session_state` flags for recording state.
   - Poll the stop condition inside the video loop and release resources when recording ends.
3. **Lip-reading import side effects**
   - Move lightweight utilities into a dedicated module (e.g., `lip_reading/utils.py`).
   - Guard training/demo setup behind `if __name__ == "__main__":` to prevent expensive work during imports.

Deliverable: both demo apps start cleanly from the repository root without manual path tweaks or hanging camera sessions.

## 2. Repository Unification

1. **Package layout**
   - Convert `Lip-Reading` and `Real-Time-Sign-Language-Translation-master` into importable packages (`lip_reading`, `sign_translator`).
   - Add `__init__.py` files and expose `predict_lip()` / `predict_asl()` helpers for external callers.
2. **Shared environment definition**
   - Consolidate dependencies into `environment.yml` (preferred for GPU workflows) or `requirements.txt` with pinned versions for TensorFlow, OpenCV, Streamlit, and audio libraries.
3. **Top-level documentation**
   - Draft `README.md` with architecture summary, setup instructions, and a quick-start guide for running each demo.

Deliverable: developers can install one environment, run either module, and import them programmatically.

## 3. Fusion Scaffolding

1. **Create integration folders**
   - Add empty `fusion_core/` with a placeholder module defining the turn-taking API contract.
   - Add `ui/` directory with a stub Streamlit entry point that will eventually orchestrate both models.
2. **Define interfaces**
   - Document expected inputs/outputs for `predict_asl()` and `predict_lip()` inside `fusion_core/README.md`.
   - Sketch event flow for alternating turns and maintaining a transcript.

Deliverable: clear placeholders showing how modules will be connected in Phase 2.

## 4. Quality and Tooling

1. **Add formatting and linting hooks**
   - Introduce `ruff` or `flake8` configuration for consistent style across packages.
2. **Basic CI scaffold**
   - Provide a GitHub Actions workflow that installs dependencies and runs static checks (unit tests can follow later).
3. **Testing strategy outline**
   - Document how to run smoke tests for each module (e.g., sample video clip inference) and note data requirements.

Deliverable: groundwork for reproducibility and future automation.

## 5. Next Steps for Phase 2

- Implement shared Streamlit UI in `ui/` to coordinate video capture for both roles.
- Add text-to-speech glue using `gTTS` or `pyttsx3` for the ASL side and a transcript viewer for lip-reading outputs.
- Evaluate latency and memory footprints before enabling simultaneous capture.

Following this plan ensures the repo meets Phase 1 goals and sets the stage for subsequent integration work.
