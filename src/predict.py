import sys, joblib, pandas as pd, pathlib

MODEL = joblib.load("data/model_gradboost.pkl")   # full Pipeline

def predict(csv_file: str):
    df = pd.read_csv(csv_file)        # raw 20-column frame
    return MODEL.predict(df)          # Pipeline handles preprocessing

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py <file.csv>")
        sys.exit(1)
    preds = predict(sys.argv[1])
    print(preds)
