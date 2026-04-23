# Churn ML Service

## Final root-level structure
- [`config/settings.py`](config/settings.py) — central settings and paths
- [`data_pipeline/dataset.py`](data_pipeline/dataset.py) — dataset loading and validation
- [`training/trainer.py`](training/trainer.py) — model training and artifact export
- [`inference/predictor.py`](inference/predictor.py) — model bundle loading and probability inference
- [`api/app.py`](api/app.py) — FastAPI application
- [`pipelines/train_pipeline.py`](pipelines/train_pipeline.py) — CLI entrypoint for training/export
- [`data`](data) — input datasets
- [`artifacts`](artifacts) — trained model and metadata
- [`Forest.ipynb`](Forest.ipynb) — research notebook using shared modules

## Workflow
1. Put the dataset into [`data/combined_users_features.csv`](data/combined_users_features.csv)
2. Train and export artifacts:
   ```bash
   ./.venv/bin/python pipelines/train_pipeline.py
   ```
3. Run the API:
   ```bash
   ./.venv/bin/uvicorn api.app:app --reload
   ```

## Artifacts
Training produces:
- [`artifacts/model.cbm`](artifacts/model.cbm)
- [`artifacts/model_metadata.json`](artifacts/model_metadata.json)

## API docs
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- OpenAPI: `http://127.0.0.1:8000/openapi.json`

## Docker
Build:
```bash
docker build -t churn-inference-service .
```

Run:
```bash
docker compose up --build
```
