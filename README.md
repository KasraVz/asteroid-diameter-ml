# Asteroid Diameter Prediction Pipeline

This project develops a machine learning pipeline to predict the diameter of asteroids based on physical and orbital parameters. The primary goal is to build a reliable model and interpret its predictions to understand which features are most indicative of an asteroid's size.

This work was completed as part of the Special Topics I course at K. N. Toosi University of Technology.

## Project Workflow

The project is structured as a series of sequential Jupyter notebooks, each responsible for a specific stage of the machine learning lifecycle.

1.  **`01_eda.ipynb` - Exploratory Data Analysis**:
    * Loads the raw dataset (`data.csv`).
    * Performs an initial inspection of data types, structure, and missing values.
    * Removes rows where the target variable (`diameter`) is missing.
    * Analyzes the distribution of the target variable, revealing a significant right-skew.
    * Saves a cleaned version of the data (`asteroids_clean.csv`) for the next stage.

2.  **`02_preprocessing.ipynb` - Feature Engineering & Preprocessing**:
    * Selects relevant features, dropping columns with excessive missing values or redundancy.
    * Builds a robust `scikit-learn` pipeline to handle data transformations.
    * The pipeline imputes missing numerical values using the median and scales them with `StandardScaler`.
    * Categorical features are imputed using the most frequent value and then one-hot encoded.
    * The data is split into training and validation sets (80/20 split).
    * The fitted preprocessing pipeline (`preprocess.pkl`) is saved to disk to ensure consistent transformations in subsequent steps.

3.  **`03_baselines.ipynb` - Baseline Models**:
    * Establishes baseline performance metrics.
    * Evaluates a `DummyRegressor` (predicting the median) to set the absolute performance floor.
    * Evaluates a `LinearRegression` model to determine if simple linear relationships exist in the data.

4.  **`04_tree_models.ipynb` - Tree-Based Ensemble Models**:
    * Trains and evaluates more advanced, non-linear models.
    * A `RandomForestRegressor` is trained with sensible default parameters.
    * A `GradientBoostingRegressor` is tuned using `RandomizedSearchCV` to find the optimal set of hyperparameters.
    * The performance of all models is compared, and the best-performing model (`model_gradboost.pkl`) is saved.

5.  **`05_interpretation.ipynb` - Model Interpretation**:
    * Loads the final tuned `GradientBoostingRegressor` model.
    * Analyzes feature importance using three methods:
        1.  Built-in impurity-based importance.
        2.  Model-agnostic Permutation Importance.
        3.  **SHAP (SHapley Additive exPlanations)** for both global and local interpretability.
    * Visualizes the results to understand the key drivers of the model's predictions.

## Setup & Usage

### Quick Start

To set up the environment and run the notebooks:

```bash
conda env create -f env.yml
conda activate asteroid-ml
jupyter lab
```

### Predict on New Data

A simple prediction script is available to use the trained model on new data.

```bash
python src/predict.py new.csv > preds.txt
```

## Results

The following table summarizes the performance of all evaluated models on the hold-out validation set. The Gradient Boosting model demonstrated superior performance across all metrics.

| Model                     | MAE (km) | RMSE (km) | R²     |
| ------------------------- | -------- | --------- | ------ |
| Dummy (median)            | 2.656    | 6.864     | -0.047 |
| Linear Regression         | 2.227    | 9.304     | -0.925 |
| Random Forest             | 0.667    | 3.065     | 0.791  |
| **GradientBoost (final)** | **0.671**| **2.776** | **0.829**|

## Key Findings

Model interpretation using SHAP revealed that the model's predictions are driven by physically meaningful features:

* **Primary Driver**: The **absolute magnitude (`H`)** is the most influential feature. Higher `H` values (fainter objects) strongly push the predicted diameter lower.
* **Secondary Driver**: The **albedo** (reflectivity) is the second most important feature. For a given brightness, higher albedo pushes the diameter prediction higher, as a shinier object must be larger to have the same apparent magnitude.
* **Minor Factors**: Orbital parameters (inclination, period, etc.) provide only marginal refinements to the predictions.

This confirms that the model has successfully learned the key physical relationships from the data rather than relying on spurious correlations.
