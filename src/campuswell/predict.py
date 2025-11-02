import joblib
import pandas as pd

def load_pipeline(path):
    return joblib.load(path)

def predict_one(pipe, record):
    df = pd.DataFrame([record])
    p = float(pipe.predict_proba(df)[:,1][0])
    return {
        "prediction": int(p >= 0.5),
        "probability": p
    }
