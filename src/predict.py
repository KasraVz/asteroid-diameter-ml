import sys, joblib, pandas as pd, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
# We **no longer** need the separate preprocess object
MODEL = joblib.load(ROOT / "data" / "model_gradboost.pkl")

def predict(csv_file: str):
    df = pd.read_csv(csv_file)
    # Just feed raw features; the Pipeline's first step will transform
    return MODEL.predict(df)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py <new_data.csv>")
        sys.exit(1)
    preds = predict(sys.argv[1])
    print(preds)
