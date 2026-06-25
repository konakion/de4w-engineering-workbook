# Project Checklist

Use this checklist before teaching or submitting the project.

## Imports

```bash
uv run python -c "import app.api, app.training, app.evaluation, app.model_utils, app.pipeline; print('final imports work')"
```

## Documentation Render

```bash
PATH=.venv/bin:$PATH quarto render docs
```

## Slide Deck Render

```bash
PATH=.venv/bin:$PATH quarto render docs/slides/de4w_capstone_project.qmd
```

## API Startup

```bash
uv run uvicorn app.api:app --reload
```

Verify:

```bash
curl http://127.0.0.1:8000/health
```

Expected:

```text
model_loaded: true
```

## Dashboard Startup

```bash
uv run streamlit run dashboard.py
```

Verify that the dashboard opens and shows the waiting state.

## phyphox Live Demo

- Phone and laptop are on the same local network.
- phyphox remote access is enabled.
- The phyphox URL opens in the laptop browser.
- If eduroam blocks traffic, use a hotspot or local router.
