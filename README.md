# 🧠 CampusWell Depression Risk Predictor

A machine learning-powered web application that assesses depression risk levels among students based on academic and lifestyle factors. Built with FastAPI, scikit-learn, and SHAP for explainable AI.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🌐 Live Demo

**Try it now**: [https://campuswell.onrender.com](https://campuswell.onrender.com)

> **Note**: The app is hosted on Render's free tier. The first request may take 30-60 seconds as the server spins up from sleep mode.

## 📋 Table of Contents

- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Deployment](#deployment)
- [Model Information](#model-information)
- [Contributing](#contributing)
- [Disclaimer](#disclaimer)

## ✨ Features

- **Interactive Web Interface**: User-friendly form for collecting student lifestyle and academic data
- **Real-time Predictions**: Instant depression risk assessment using trained ML model
- **Explainable AI**: SHAP (SHapley Additive exPlanations) visualizations showing which factors influenced the prediction
- **Smart Feature Aggregation**: Clean, non-duplicated feature importance display
- **Probability Capping**: Risk probabilities capped at 95% to reflect model uncertainty
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **RESTful API**: Programmatic access to predictions via JSON API
- **Health Monitoring**: Built-in health check endpoint for deployment monitoring

## 🎯 Demo

### 🌐 Live Application
Visit the deployed application: **[https://campuswell.onrender.com](https://campuswell.onrender.com)**

### Web Interface
1. **Landing Page**: Welcome screen with clear call-to-action
2. **Assessment Form**: Comprehensive questionnaire covering:
   - Demographics (Gender, Age)
   - Academic factors (Pressure, Study Hours, Satisfaction)
   - Lifestyle (Sleep Duration, Dietary Habits)
   - Mental health indicators (Suicidal Thoughts, Family History)
   - Financial stress levels
3. **Results Page**: 
   - Risk classification (High/Low Risk)
   - Probability percentage
   - SHAP feature importance visualization
   - Complete input summary
   - Mental health resources

### API Endpoint

**Live API**: `https://campuswell.onrender.com/predict`

```bash
curl -X POST "https://campuswell.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Gender": "Male",
    "Age": 20,
    "Academic Pressure": 4,
    ...
  }'
```

**Local Development**: `http://127.0.0.1:8000/predict`

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Gender": "Male",
    "Age": 20,
    "Academic Pressure": 4,
    ...
  }'
```

## 🚀 Quick Start (Windows Users)

For the fastest setup on Windows:

1. **Clone and navigate**
   ```bash
   git clone https://github.com/zhengyaohuang/campuswell.git
   cd campuswell
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Install dependencies**
   ```bash
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   start_app.bat
   ```
   
   The browser will automatically open to http://127.0.0.1:8000 🎉

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/zhengyaohuang/campuswell.git
cd campuswell
```

2. **Create virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Prepare the model**
- Place your trained model file at: `models/best_pipeline.joblib`
- (Optional) Add `models/metadata.json` for model metadata
- Ensure `Depression Student Dataset.csv` is in the root directory

5. **Run the application**

**Option A: Quick Start (Windows)**
```bash
# Double-click or run from command prompt
start_app.bat
```
This will:
- Activate the virtual environment
- Set up the Python path
- Start the FastAPI server
- Automatically open http://127.0.0.1:8000 in your browser

**Option B: Manual Start (All platforms)**
```bash
# Windows
set PYTHONPATH=src
python -m uvicorn app.main:app --reload

# macOS/Linux
export PYTHONPATH=src
python -m uvicorn app.main:app --reload
```

6. **Access the application**
- Web Interface: http://127.0.0.1:8000/
- Assessment Form: http://127.0.0.1:8000/form
- API Documentation: http://127.0.0.1:8000/docs
- Health Check: http://127.0.0.1:8000/health

## 📖 Usage

### Web Interface

1. Navigate to http://127.0.0.1:8000/
2. Click "Start Prediction"
3. Fill out the assessment form:
   - Select your demographic information
   - Rate academic pressure and satisfaction (1-5 scale)
   - Provide lifestyle information
   - Answer mental health questions
4. Submit the form to receive:
   - Depression risk classification
   - Probability score
   - SHAP explanation showing which factors influenced your result
   - Helpful mental health resources

### API Usage

**Endpoint**: `POST /predict`

**Request Body**:
```json
{
  "Gender": "Male",
  "Age": 20,
  "Academic Pressure": 4,
  "Study Satisfaction": 3,
  "Sleep Duration": "5-6 hours",
  "Dietary Habits": "Moderate",
  "Have you ever had suicidal thoughts ?": "No",
  "Study Hours": 8,
  "Financial Stress": 3,
  "Family History of Mental Illness": "No"
}
```

**Response**:
```json
{
  "prediction": "Low Risk",
  "probability": 0.23,
  "features_used": [...]
}
```

## 📁 Project Structure

```
campuswell/
├── app/
│   ├── main.py                 # FastAPI application
│   ├── static/
│   │   ├── style.css          # Styling
│   │   └── shap_plot.png      # Generated SHAP plots
│   └── templates/
│       ├── index.html         # Landing page
│       ├── form.html          # Assessment form
│       └── result.html        # Results display
├── models/
│   ├── best_pipeline.joblib   # Trained ML model
│   └── metadata.json          # Model metadata
├── src/
│   └── campuswell/
│       └── predict.py         # Prediction utilities
├── Depression Student Dataset.csv  # Training data
├── requirements.txt           # Python dependencies
├── start_app.bat             # 🪟 Windows quick start script
├── start_web.bat             # Alternative Windows script
└── README.md                 # This file
```

## 📊 API Documentation

FastAPI automatically generates interactive API documentation:

- **Live Swagger UI**: https://campuswell.onrender.com/docs
- **Live ReDoc**: https://campuswell.onrender.com/redoc
- **Local Swagger UI**: http://127.0.0.1:8000/docs (when running locally)
- **Local ReDoc**: http://127.0.0.1:8000/redoc (when running locally)

### Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Landing page |
| `/form` | GET | Assessment form |
| `/predict_form` | POST | Submit form prediction |
| `/predict` | POST | JSON API prediction |
| `/health` | GET | Health check |
| `/meta` | GET | Model metadata |

## 🌐 Deployment

### Render Deployment

1. **Create `render.yaml`** in root directory:
```yaml
services:
  - type: web
    name: campuswell-predictor
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn app.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
      - key: PYTHONPATH
        value: src
```

2. **Push to GitHub**
```bash
git add .
git commit -m "Deploy to Render"
git push origin main
```

3. **Connect on Render**
- Go to [render.com](https://render.com)
- Create new Web Service
- Connect your GitHub repository
- Render will auto-detect `render.yaml`
- Click "Create Web Service"

### Environment Variables

Set these in your deployment platform:
- `PYTHONPATH=src` (Required)
- `PORT` (Auto-set by most platforms)

### Docker Deployment (Optional)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
ENV PYTHONPATH=src
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🤖 Model Information

### Model Pipeline
- **Preprocessing**: ColumnTransformer with categorical and numerical transformers
- **Algorithm**: Logistic Regression (binary classification)
- **Features**: 10 input features including demographics, academic, and lifestyle factors
- **Target**: Depression (Yes/No)

### Performance
- Probabilities capped at 95% to account for model uncertainty
- SHAP values computed using LinearExplainer for interpretability
- Feature importance aggregated for one-hot encoded variables

### Training Data
- Dataset: `Depression Student Dataset.csv`
- Size: 500 student records
- Features: Gender, Age, Academic Pressure, Study Satisfaction, Sleep Duration, Dietary Habits, Suicidal Thoughts, Study Hours, Financial Stress, Family History

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest

# Format code
black app/

# Lint code
flake8 app/
```

## ⚠️ Disclaimer

**IMPORTANT**: This application is for **educational and research purposes only**. 

- **Not a Medical Diagnosis**: This tool cannot diagnose depression or any mental health condition
- **Not a Substitute for Professional Help**: Always consult qualified healthcare professionals for mental health concerns
- **Statistical Model**: Predictions are based on statistical patterns and may not apply to individual cases
- **Probability Limitations**: Even high probabilities do not guarantee outcomes
- **Seek Help**: If you're experiencing mental health issues, please contact:
  - **National Suicide Prevention Lifeline**: 988 (US)
  - **Crisis Text Line**: Text HOME to 741741
  - **International**: Find resources at [findahelpline.com](https://findahelpline.com)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Zhengyao Huang** - [GitHub](https://github.com/zhengyaohuang)

## 🙏 Acknowledgments

- Depression Student Dataset contributors
- FastAPI framework developers
- SHAP library for explainable AI
- Anthropic Claude for development assistance
- Mental health professionals who provided domain expertise

## 📞 Support

For questions or issues:
- Open an issue on [GitHub](https://github.com/zhengyaohuang/campuswell/issues)
- Visit the live app: [https://campuswell.onrender.com](https://campuswell.onrender.com)

---

**Made with ❤️ for student mental health awareness**

**Live at**: [campuswell.onrender.com](https://campuswell.onrender.com)