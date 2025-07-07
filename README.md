# Asteroid Diameter ML Pipeline

## Quick start
conda env create -f env.yml
conda activate asteroid-ml
jupyter lab                     # open notebooks

## Predict on new data
python src/predict.py new.csv > preds.txt

## Final metrics (validation set)
Model | MAE | RMSE | R²
----- | --- | ---- | ---
GradientBoost (final) | 1.15 km | 2.34 km | 0.82
