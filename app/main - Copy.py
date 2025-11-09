from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Extra
import os, json

import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from campuswell.predict import load_pipeline, predict_one


# ===============================
# PATH SETUP
# ===============================
BASE = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE, "models", "best_pipeline.joblib")
META_PATH = os.path.join(BASE, "models", "metadata.json")

app = FastAPI(title="Depression Risk Predictor")

# Serve static and templates
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))


class Input(BaseModel, extra=Extra.allow):
    pass


# ===============================
# LOAD MODEL
# ===============================
@app.on_event("startup")
def _load_model():
    app.state.model = load_pipeline(MODEL_PATH)
    if os.path.exists(META_PATH):
        with open(META_PATH, "r") as f:
            app.state.meta = json.load(f)
    else:
        app.state.meta = {}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/meta")
def meta():
    return app.state.meta


# ===============================
# API ENDPOINT
# ===============================
@app.post("/predict")
def predict_api(x: Input):
    return predict_one(app.state.model, x.dict())


# ===============================
# WEB UI ROUTES
# ===============================
@app.get("/", response_class=HTMLResponse)
def landing_page(request: Request):
    """Landing page for the web app"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/form", response_class=HTMLResponse)
def show_form(request: Request):
    """Prediction form page"""
    return templates.TemplateResponse("form.html", {"request": request})


@app.post("/predict_form", response_class=HTMLResponse)
def predict_form(
    request: Request,
    Gender: str = Form(...),
    Age: int = Form(...),
    Academic_Pressure: int = Form(..., alias="Academic Pressure"),
    Study_Satisfaction: int = Form(..., alias="Study Satisfaction"),
    Sleep_Duration: str = Form(..., alias="Sleep Duration"),
    Dietary_Habits: str = Form(..., alias="Dietary Habits"),
    Suicidal_Thoughts: str = Form(..., alias="Have you ever had suicidal thoughts ?"),
    Study_Hours: int = Form(..., alias="Study Hours"),
    Financial_Stress: int = Form(..., alias="Financial Stress"),
    Family_History: str = Form(..., alias="Family History of Mental Illness"),
):
    # ===============================
    # USER INPUT
    # ===============================
    record = {
        "Gender": Gender,
        "Age": Age,
        "Academic Pressure": Academic_Pressure,
        "Study Satisfaction": Study_Satisfaction,
        "Sleep Duration": Sleep_Duration,
        "Dietary Habits": Dietary_Habits,
        "Have you ever had suicidal thoughts ?": Suicidal_Thoughts,
        "Study Hours": Study_Hours,
        "Financial Stress": Financial_Stress,
        "Family History of Mental Illness": Family_History,
    }

    model = app.state.model
    df = pd.DataFrame([record])
    proba = float(model.predict_proba(df)[:, 1][0])
    label = "High Risk of Depression" if proba >= 0.5 else "Low Risk of Depression"

    # ===============================
    # SHAP EXPLANATION (KernelExplainer)
    # ===============================
    try:
        preprocessor = model.named_steps["preprocessor"]
        clf = model.named_steps["model"]

        # Background data for SHAP comparison
        background_df = pd.DataFrame([
            record,
            record.copy(),
            {
                "Gender": "Female",
                "Age": 21,
                "Academic Pressure": 3,
                "Study Satisfaction": 3,
                "Sleep Duration": "7-8 hours",
                "Dietary Habits": "Moderate",
                "Have you ever had suicidal thoughts ?": "No",
                "Study Hours": 5,
                "Financial Stress": 2,
                "Family History of Mental Illness": "No"
            }
        ])

        # Transform data
        X_bg = preprocessor.transform(background_df)
        X_in = preprocessor.transform(df)

        # Convert sparse → dense if needed
        if hasattr(X_bg, "toarray"):
            X_bg = X_bg.toarray()
        if hasattr(X_in, "toarray"):
            X_in = X_in.toarray()

        # Get feature names
        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            feature_names = [f"feature_{i}" for i in range(X_in.shape[1])]

        # Define SHAP function for Logistic Regression
        f = lambda x: clf.predict_proba(x)[:, 1]

        # KernelExplainer (model-agnostic)
        explainer = shap.KernelExplainer(f, X_bg)
        shap_values = explainer.shap_values(X_in, nsamples=100)

        # Generate bar plot
        plt.figure()
        shap.summary_plot(
            shap_values,
            X_in,
            feature_names=feature_names,
            plot_type="bar",
            show=False
        )
        plt.tight_layout()

        shap_path = os.path.join(os.path.dirname(__file__), "static", "shap_plot.png")
        plt.savefig(shap_path)
        plt.close()
        shap_available = True

    except Exception as e:
        shap_available = False
        print("⚠️ SHAP explanation failed:", e)

    # ===============================
    # RETURN RESULT PAGE
    # ===============================
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "label": label,
            "prob": round(proba * 100, 2),
            "shap_available": shap_available,
            "shap_img": "/static/shap_plot.png" if shap_available else None,
        },
    )
