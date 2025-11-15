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

# Assuming 'campuswell.predict' is available in your environment
from campuswell.predict import load_pipeline, predict_one 


# ===============================
# PATH SETUP
# ===============================
BASE = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE, "models", "best_pipeline.joblib")
META_PATH = os.path.join(BASE, "models", "metadata.json")
# Path to the uploaded dataset
DATASET_PATH = "Depression Student Dataset.csv"

app = FastAPI(title="Depression Risk Predictor")

# Serve static and templates
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))


class Input(BaseModel, extra=Extra.allow):
    pass


# ===============================
# LOAD MODEL AND DATA (for SHAP background)
# ===============================
@app.on_event("startup")
def _load_model():
    # Load the machine learning pipeline
    app.state.model = load_pipeline(MODEL_PATH)
    print(f"✅ Model loaded: {type(app.state.model)}")
    print(f"✅ Pipeline steps: {app.state.model.named_steps.keys()}")
    
    # Load metadata
    if os.path.exists(META_PATH):
        with open(META_PATH, "r") as f:
            app.state.meta = json.load(f)
    else:
        app.state.meta = {}

    # Load and preprocess data for SHAP background
    try:
        full_data = pd.read_csv(DATASET_PATH)
        # Drop the target variable (Depression)
        app.state.shap_background_data = full_data.drop(columns=["Depression"], errors="ignore")
        print(f"✅ Loaded {len(app.state.shap_background_data)} records for SHAP background")
        print(f"✅ Columns: {list(app.state.shap_background_data.columns)}")
    except Exception as e:
        print(f"⚠️ Failed to load dataset for SHAP background: {e}")
        app.state.shap_background_data = None


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
    
    # Make prediction
    try:
        proba = float(model.predict_proba(df)[:, 1][0])
        
        # Cap probability at 95% to reflect model uncertainty
        # No model is 100% certain
        display_proba = min(proba, 0.95)
        
        label = "High Risk of Depression" if proba >= 0.5 else "Low Risk of Depression"
        confidence = "high" if proba >= 0.7 or proba <= 0.3 else "moderate"
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "label": "Error in prediction",
                "prob": 0,
                "confidence": "error",
                "user_inputs": record,
                "shap_available": False,
                "shap_img": None,
                "error_message": str(e),
            },
        )

    # ===============================================================
    # SHAP EXPLANATION (COMPREHENSIVE DEBUG VERSION)
    # ===============================================================
    shap_available = False
    error_message = ""
    
    if app.state.shap_background_data is None:
        error_message = "Background data not loaded"
        print(f"❌ {error_message}")
    else:
        try:
            print("\n" + "="*60)
            print("🔍 STARTING SHAP EXPLANATION")
            print("="*60)
            
            # Get preprocessor and classifier from pipeline
            try:
                preprocessor = model.named_steps["preprocess"]
                print(f"✅ Preprocessor type: {type(preprocessor)}")
            except KeyError:
                print(f"❌ Available steps: {list(model.named_steps.keys())}")
                raise Exception("No 'preprocess' step found in pipeline")
            
            try:
                clf = model.named_steps["model"]
                print(f"✅ Classifier type: {type(clf).__name__}")
            except KeyError:
                # Try alternative names
                for step_name in ['classifier', 'clf', 'estimator']:
                    if step_name in model.named_steps:
                        clf = model.named_steps[step_name]
                        print(f"✅ Classifier found as '{step_name}': {type(clf).__name__}")
                        break
                else:
                    print(f"❌ Available steps: {list(model.named_steps.keys())}")
                    raise Exception("No classifier found in pipeline")
            
            # Sample background data
            n_samples = min(50, len(app.state.shap_background_data))
            background_df = app.state.shap_background_data.sample(n=n_samples, random_state=42)
            print(f"✅ Background sample size: {n_samples}")
            
            # Transform data
            print("🔄 Transforming background data...")
            X_bg = preprocessor.transform(background_df)
            print(f"   Shape after transform: {X_bg.shape}")
            print(f"   Type: {type(X_bg)}")
            print(f"   Is sparse: {hasattr(X_bg, 'toarray')}")
            
            print("🔄 Transforming input data...")
            X_in = preprocessor.transform(df)
            print(f"   Shape after transform: {X_in.shape}")
            print(f"   Type: {type(X_in)}")
            
            # Convert sparse matrices to dense
            if hasattr(X_bg, 'toarray'):
                print("🔄 Converting background to dense array...")
                X_bg = X_bg.toarray()
            if hasattr(X_in, 'toarray'):
                print("🔄 Converting input to dense array...")
                X_in = X_in.toarray()
            
            print(f"✅ Final shapes - Background: {X_bg.shape}, Input: {X_in.shape}")
            
            # Get feature names
            try:
                feature_names = list(preprocessor.get_feature_names_out())
                print(f"✅ Got {len(feature_names)} feature names using get_feature_names_out()")
            except AttributeError:
                print("⚠️ get_feature_names_out() not available, using generic names")
                feature_names = [f"feature_{i}" for i in range(X_in.shape[1])]
            
            # Determine the best explainer
            model_type = type(clf).__name__
            print(f"\n🤖 Model type detected: {model_type}")
            
            # Test if model has predict_proba
            try:
                test_proba = clf.predict_proba(X_in)
                print(f"✅ Model has predict_proba: output shape {test_proba.shape}")
            except Exception as e:
                print(f"⚠️ predict_proba test failed: {e}")
            
            print("\n🔧 Creating SHAP explainer...")
            
            # Try different explainers based on model type
            if 'Logistic' in model_type or 'Linear' in model_type:
                print("   Using LinearExplainer...")
                explainer = shap.LinearExplainer(clf, X_bg, feature_perturbation="interventional")
                shap_values = explainer.shap_values(X_in)
                
                # Handle multi-output for binary classification
                if isinstance(shap_values, list):
                    print(f"   Got list of {len(shap_values)} arrays")
                    shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
                
            elif 'Tree' in model_type or 'Forest' in model_type or 'Boost' in model_type or 'XGB' in model_type:
                print("   Using TreeExplainer...")
                explainer = shap.TreeExplainer(clf)
                shap_values = explainer.shap_values(X_in)
                
                # Handle multi-output for binary classification
                if isinstance(shap_values, list):
                    print(f"   Got list of {len(shap_values)} arrays")
                    shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
                    
            else:
                print("   Using KernelExplainer (slower but universal)...")
                def predict_fn(x):
                    return clf.predict_proba(x)[:, 1]
                
                explainer = shap.KernelExplainer(predict_fn, X_bg[:20])  # Use fewer samples for speed
                shap_values = explainer.shap_values(X_in, nsamples=50)
            
            print(f"✅ SHAP values computed!")
            print(f"   Type: {type(shap_values)}")
            print(f"   Shape: {np.array(shap_values).shape}")
            
            # Flatten if needed
            if len(np.array(shap_values).shape) > 1:
                shap_values = shap_values[0]
                print(f"   Flattened to shape: {np.array(shap_values).shape}")
            
            # Create visualization showing direction of impact
            print("\n📊 Creating visualization...")
            
            # Clean up feature names FIRST
            cleaned_names = []
            for name in feature_names:
                # Remove common prefixes
                clean_name = name.replace('num_', '').replace('cat_', '').replace('remainder__', '')
                clean_name = clean_name.replace('onehotencoder__', '').replace('ordinalencoder__', '')
                # Remove leading/trailing underscores
                clean_name = clean_name.strip('_')
                
                # For categorical features, remove the category value suffix
                # This aggregates all categories of the same feature into one
                if '_' in clean_name:
                    # Common patterns: feature_name_category_value
                    # We want to keep only the feature_name part
                    parts = clean_name.split('_')
                    
                    # Check if last part looks like a category value
                    if len(parts) >= 2:
                        last_part = parts[-1].lower()
                        # List of common category indicators
                        category_keywords = ['healthy', 'unhealthy', 'moderate', 'poor', 'yes', 'no', 
                                           'male', 'female', 'hours', 'than', 'very', 'high', 'low',
                                           'less', 'more', 'satisfied', 'dissatisfied']
                        
                        # If last part contains category keywords or starts with lowercase, remove it
                        if any(keyword in last_part for keyword in category_keywords) or last_part[0].islower():
                            clean_name = '_'.join(parts[:-1])
                
                # Replace remaining underscores with spaces for readability
                clean_name = clean_name.replace('_', ' ')
                cleaned_names.append(clean_name)
            
            print(f"📝 Cleaned feature names sample: {cleaned_names[:5]}")
            
            # Aggregate SHAP values by cleaned feature name (keeping sign for direction)
            feature_shap_dict = {}
            for clean_name, shap_val in zip(cleaned_names, shap_values):
                if clean_name in feature_shap_dict:
                    # Sum SHAP values to aggregate multiple encodings of same feature
                    feature_shap_dict[clean_name] += shap_val
                else:
                    feature_shap_dict[clean_name] = shap_val
            
            print(f"📊 Aggregated features count: {len(feature_shap_dict)}")
            print(f"📊 Top aggregated features: {list(feature_shap_dict.keys())[:10]}")
            
            # Sort by absolute importance but keep the sign
            sorted_features = sorted(feature_shap_dict.items(), 
                                   key=lambda x: abs(x[1]), 
                                   reverse=True)
            n_display = min(10, len(sorted_features))
            top_features = sorted_features[:n_display]
            
            # Reverse for plotting (highest at top)
            top_features = list(reversed(top_features))
            
            display_names = [name for name, _ in top_features]
            display_values = [value for _, value in top_features]
            
            # Truncate long names
            display_names = [name[:45] + '...' if len(name) > 45 else name for name in display_names]
            
            print(f"📊 Final display features: {display_names}")
            
            # Create plot with color-coded direction
            plt.figure(figsize=(10, 6))
            
            # Color bars based on direction: red for positive (increases risk), blue for negative (decreases risk)
            colors = ['#ff6b6b' if val > 0 else '#4dabf7' for val in display_values]
            
            plt.barh(range(n_display), display_values, color=colors)
            plt.yticks(range(n_display), display_names, fontsize=9)
            
            # Add a vertical line at zero
            plt.axvline(x=0, color='white', linestyle='-', linewidth=1, alpha=0.5)
            
            plt.xlabel('Impact on Depression Risk', fontsize=10)
            plt.title('Top 10 Factors That Influenced Your Prediction\n(Red = Increases Risk | Blue = Decreases Risk)', fontsize=11)
            plt.tight_layout()
            
            # Save plot
            shap_path = os.path.join(static_dir, "shap_plot.png")
            plt.savefig(shap_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Plot saved to: {shap_path}")
            print(f"✅ File exists: {os.path.exists(shap_path)}")
            
            shap_available = True
            print("\n" + "="*60)
            print("✅ SHAP EXPLANATION COMPLETED SUCCESSFULLY!")
            print("="*60 + "\n")

        except Exception as e:
            shap_available = False
            error_message = str(e)
            print(f"\n❌ SHAP EXPLANATION FAILED!")
            print(f"❌ Error type: {type(e).__name__}")
            print(f"❌ Error message: {error_message}")
            print("\n📋 Full traceback:")
            import traceback
            traceback.print_exc()
            print("="*60 + "\n")

    # Create readable labels for scales
    scale_labels = {
        1: "Very Low", 2: "Low", 3: "Moderate", 4: "High", 5: "Very High"
    }
    satisfaction_labels = {
        1: "Very Dissatisfied", 2: "Dissatisfied", 3: "Neutral", 
        4: "Satisfied", 5: "Very Satisfied"
    }
    
    # Format user inputs for display
    display_inputs = {
        "Gender": Gender,
        "Age": f"{Age} years old",
        "Academic Pressure": f"{Academic_Pressure} ({scale_labels.get(Academic_Pressure, '')})",
        "Study Satisfaction": f"{Study_Satisfaction} ({satisfaction_labels.get(Study_Satisfaction, '')})",
        "Sleep Duration": Sleep_Duration,
        "Dietary Habits": Dietary_Habits,
        "Suicidal Thoughts": Suicidal_Thoughts,
        "Study Hours": f"{Study_Hours} hours per day",
        "Financial Stress": f"{Financial_Stress} ({scale_labels.get(Financial_Stress, '')})",
        "Family History": Family_History,
    }

    # ===============================
    # RETURN RESULT PAGE
    # ===============================
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "label": label,
            "prob": round(display_proba * 100, 1),
            "confidence": confidence,
            "user_inputs": display_inputs,
            "shap_available": shap_available,
            "shap_img": "/static/shap_plot.png" if shap_available else None,
            "shap_error": error_message if not shap_available and error_message else None,
        },
    )