RainShield AI — Backend Connected Prototype

1. Start the FastAPI backend:
   cd backend
   py -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   uvicorn app:app --reload --port 8000

2. Open frontend/index.html in Chrome.

3. Select Assam/Delhi and 1h/3h/6h. The browser calls:
   POST http://127.0.0.1:8000/predict

IMPORTANT:
The backend currently contains deterministic DEMO predictions because no trained
XGBoost model or training dataset was supplied with the project. The API contract
is ready for a real model. Replace the DEMO logic in backend/app.py with
model.predict(...) / model.predict_proba(...) once the trained model is available.
