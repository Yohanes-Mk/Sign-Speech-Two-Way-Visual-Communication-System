# Sign-Speech Phase 1 Mission Plan

> Phase 1 is where both prototypes relearn how to breathe together. We’re keeping the magic real-time while building the scaffolding for a shared conversation loop.

---

## Phase Snapshot
- 🎯 **Current phase:** Phase 1 – Module Stabilization (in progress)
- ✅ **Phase 0:** Vision & documentation story published
- ⏭️ **Next horizon:** Phase 2 – Streamlit fusion interface

---

## Pillar 1 — Stabilize the Vision-Only Demos
- [ ] Harden model asset loading in `Real-Time-Sign-Language-Translation-master/app.py` using `Path(__file__)` helpers so weights travel with the package.
- [ ] Manage Streamlit camera lifecycles with `st.session_state` flags and explicit resource teardown to prevent zombie capture loops.
- [ ] Refactor lip-reading helper imports into `lip_reading/utils.py` and wrap heavy training code in `if __name__ == "__main__":` guards.

**Deliverable:** Both demos spin up from the repo root, infer on sample clips, and shut down gracefully—no manual path edits, no hanging webcam sessions.

---

## Pillar 2 — Unify the Repository Skeleton
- [ ] Promote `Lip-Reading/` and `Real-Time-Sign-Language-Translation-master/` into installable packages (`lip_reading`, `sign_translator`) with clean `predict_lip()` / `predict_asl()` entry points.
- [ ] Author a single environment manifest (`environment.yml` preferred; fallback: consolidated `requirements.txt`) pinned for TensorFlow, MediaPipe, OpenCV, Streamlit, and audio stack.
- [ ] Keep top-level docs in sync (README + module READMEs) so newcomers can install once and launch either experience.

**Deliverable:** One environment, two importable modules, zero confusion. Running either demo or calling the APIs becomes a copy-paste exercise.

---

## Pillar 3 — Lay Fusion Scaffolding
- [ ] Create `fusion_core/` with placeholder orchestrator contract (request/response objects, latency notes, turn-taking API).
- [ ] Seed `ui/` with a Streamlit stub that will later juggle both video feeds and transcripts.
- [ ] Document interface expectations and event flow inside `fusion_core/README.md`.

**Deliverable:** Visible anchor points that make Phase 2’s integration work obvious and inviting.

---

## Pillar 4 — Quality, Tooling, and Signals
- [ ] Add repo-wide linting/formatting (e.g., Ruff or Flake8 + black) with lightweight pre-commit hooks.
- [ ] Stand up a GitHub Actions workflow that installs the environment and runs static checks.
- [ ] Outline smoke-test recipes (sample sign clip, lip-reading clip) and note dataset licensing boundaries.

**Deliverable:** Baseline automation that keeps the research tempo high without sacrificing reproducibility.

---

## Phase 2 Preview — What’s Waiting on Deck
- Build the shared Streamlit fusion UI that conducts signer ↔ speaker turn-taking.
- Wire text-to-speech feedback and transcript viewers for both modalities.
- Benchmark latency, WER, and confidence arbitration before simultaneous capture goes live.

Stay in Phase 1 until the checklist above feels boring—then we graduate to orchestration mode.
