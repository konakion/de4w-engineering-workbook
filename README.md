# DE4W Live Activity Recognition App

Educational capstone project for the course **Data Engineering for Wearables**.

The project demonstrates how to turn wearable sensor data into a working activity recognition system:

```text
WISDM accelerometer data
→ windowing
→ feature engineering
→ Random Forest
→ evaluation
→ FastAPI
→ Streamlit
→ phyphox live prediction
```

## Educational Purpose

Students learn how the pieces of a wearable ML system fit together:

- loading raw sensor files,
- segmenting streams into windows,
- extracting statistical features,
- training a baseline model,
- evaluating generalization to unseen people,
- exporting model artifacts,
- serving predictions with FastAPI,
- running a live dashboard with phyphox.

The goal is not to build the most complex model.

The goal is to understand the full engineering pipeline.

## Tooling In Plain Language

- `uv` creates and uses the project environment so everyone runs the same dependencies.
- FastAPI exposes the trained Model through web endpoints such as `/health` and `/predict`.
- `uvicorn` is the web server that runs the FastAPI app on your machine.
- Streamlit turns Python code into a browser dashboard for teaching and live demos.
- `joblib` saves and loads scikit-learn model artifacts.
- The `app/` folder is a Python package, which keeps reusable project code separate from notebooks and docs.

## Quick Start

Install dependencies:

```bash
uv sync
```

`uv sync` installs the dependencies defined by this project into a local environment.

Check imports:

```bash
uv run python -c "import app.api, app.training, app.evaluation, app.model_utils, app.pipeline; print('final imports work')"
```

Render the Quarto book:

```bash
PATH=.venv/bin:$PATH quarto render docs
```

Start the API:

```bash
uv run uvicorn app.api:app --reload
```

`uv run` executes the command inside the project environment, and `uvicorn` serves the FastAPI app.

Start the dashboard:

```bash
uv run streamlit run dashboard.py
```

Streamlit opens a browser UI for the live demo without requiring a separate frontend project.

The dashboard includes Live phyphox mode and Replay Mode. Use Replay Mode when
networking or phyphox is unavailable during a class demo.

## Repository Structure

```text
app/
  api.py                 FastAPI endpoints
  config.py              shared paths and defaults
  data_loader.py         WISDM loading
  evaluation.py          random, subject-wise and LOSO evaluation
  feature_extraction.py  window feature extraction
  live_buffer.py         live sample buffer
  model_utils.py         prediction service
  phyphox_client.py      phyphox HTTP client
  pipeline.py            reusable feature table pipeline
  training.py            model training and artifact export

data/                    WISDM raw data and feature table
models/                  trained model artifacts and metadata
docs/                    Quarto book, slides and teaching guides
dashboard.py             Streamlit live demo
```

## Data Placement

Place WISDM phone accelerometer files in `data/`.

Expected file names look like:

```text
data_1600_accel_phone.txt
data_1601_accel_phone.txt
```

The project focuses on:

- walking
- sitting
- standing

## Core Commands

Render docs:

```bash
PATH=.venv/bin:$PATH quarto render docs
```

Render slides:

```bash
PATH=.venv/bin:$PATH quarto render docs/slides/de4w_capstone_project.qmd
```

Check final imports:

```bash
uv run python -c "import app.api, app.training, app.evaluation, app.model_utils, app.pipeline; print('final imports work')"
```

Start FastAPI:

```bash
uv run uvicorn app.api:app --reload
```

Check API health:

```bash
curl http://127.0.0.1:8000/health
```

Start Streamlit:

```bash
uv run streamlit run dashboard.py
```

## Model Artifacts

Expected files:

```text
models/activity_model_random_forest.joblib
models/feature_columns.joblib
models/activity_labels.joblib
models/model_metadata.json
```

`feature_columns.joblib` is important because live prediction must use the same feature names and order as training.

## phyphox Live Demo

On the phone:

1. Open phyphox.
2. Start an acceleration experiment.
3. Enable remote access.
4. Copy the URL shown by phyphox.
5. Paste the URL into the Streamlit dashboard.

The live path is:

```text
phyphox
→ PhyphoxClient
→ LiveBuffer
→ Window
→ Feature Vector
→ Model
→ dashboard
```

If phyphox is unavailable, select Replay Mode in the dashboard sidebar. Replay
Mode uses stored/synthetic samples to demonstrate the same Window -> Feature
Vector -> Model -> Prediction flow without a live phone.

## eduroam And Hotspot Note

The phone and laptop must communicate directly.

Some university networks, especially eduroam, block device-to-device traffic.

If the phyphox URL does not open in the laptop browser, use:

- a smartphone hotspot,
- a local WiFi router,
- a travel router.

This is an expected deployment lesson, not a model bug.

## Teaching Materials

- Quarto book: `docs/`
- Slide deck: `docs/slides/de4w_capstone_project.qmd`
- Instructor guide: `docs/instructor_guide.qmd`
- Student lab guide: `docs/lab_guide.qmd`
- Project checklist: `docs/project_checklist.md`

## Known Issues

- `quarto render docs` may use a system Python without Jupyter/PyYAML. Use `PATH=.venv/bin:$PATH quarto render docs`.
- Live prediction depends on local network reachability between phone and laptop.
- The model is a teaching baseline, not a production medical model.
