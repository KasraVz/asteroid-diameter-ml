import sys, joblib, pandas as pd, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]   # project root
PREP  = joblib.load(ROOT / "data" / "preprocess.pkl")
MODEL = joblib.load(ROOT / "data" / "model_gradboost.pkl")

def predict(csv_file):
    df = pd.read_csv(csv_file)
    X = PREP.transform(df)
    return MODEL.predict(X)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py <file.csv>")
        sys.exit(1)
    print(predict(sys.argv[1]))
