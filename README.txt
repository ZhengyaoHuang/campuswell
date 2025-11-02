HOW TO RUN (local demo)

1. Put your trained model file at:
   models/best_pipeline.joblib
   (This is the same model you already trained and used with /predict.)

2. (Optional) Put metadata.json in models/ if you have it.

3. Create & activate a virtualenv (Windows cmd):
   python -m venv venv
   venv\Scripts\activate

4. Install requirements:
   pip install -r requirements.txt

5. Run the app:
   set PYTHONPATH=src
   python -m uvicorn app.main:app --reload

6. Open in browser:
   http://127.0.0.1:8000/form   -> user-friendly form
   http://127.0.0.1:8000/docs   -> auto API docs
   http://127.0.0.1:8000/health -> status check

SHAP EXPLANATION
After you submit the form, the app will:
- Run a prediction
- Generate a bar chart of top features for that prediction
- Save it to app/static/shap_plot.png
- Show it on the result page

If SHAP fails, the page will still show prediction and probability, just without the chart.
