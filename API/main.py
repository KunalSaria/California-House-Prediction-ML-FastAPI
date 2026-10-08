import joblib
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel,Field
from fastapi.responses import StreamingResponse
import uvicorn
import io
app= FastAPI()

model = joblib.load("house_model.joblib")
features= joblib.load("house_features.joblib")

class HouseFeatures(BaseModel):
    MedInc       : float=Field(gt=0,description="Media Income of Neighbourhood"),
    HouseAge     : float=Field(gt=0,description="Average age of HouseAGE"),
    AveRooms     : float=Field(gt=0,description="AveRooms"),
    AveBedrms    : float=Field(gt=0,description="AveBedrms"),
    Population   : float=Field(gt=0,description="Population"),
    AveOccup     : float=Field(gt=0,description="AveOccup"),
    Latitude     : float=Field(gt=32,le=42 ,description="Latitude"),
    Longitude    : float=Field(gt=-125,le =-114 , description="Longitude")


@app.get("/")
def home():
    return{
        "mesaage":"California house prediction",
        "status":"running",
        "endpoint":"Send POST request to /predict"
    }

@app.get("/health")
def health():
    return{
        "status":"running",
        "model":"RandomForestRegressor",
        "Features":features,
        "avg_error":"$39,000"
    }


@app.post("/predict")
def predict(house:HouseFeatures):
    try:
        input_data= pd.DataFrame([{
            "MedInc"    : house.MedInc,
            "HouseAge"  : house.HouseAge,
            "AveRooms"  : house.AveRooms,
            "AveBedrms" : house.AveBedrms,
            "Population": house.Population,
            "AveOccup"  : house.AveOccup,
            "Latitude"  : house.Latitude,
            "Longitude" : house.Longitude}])
        predicted = model.predict(input_data)[0]
        price_usd=predicted*100000

        return{
            "Predicted_Price":f"${price_usd:,.0f}",
            "Predicted_price_short":f"${predicted:.2f} hundred thousand",
            "fidence_range": f"${price_usd-39000:,.2f} to ${price_usd:,.2f}"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            delail=f"prediction failed:{str(e)}"
        )



@app.post('/predict-file')
async def predict_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code= 400,
            detail="please upload a CSV file"
        )
    contents = await file.read()
    df=pd.read_csv(io.BytesIO(contents))
    print("ACTUAL COLUMNS:", df.columns.tolist())
    df.columns = df.columns.str.strip()

    required_coln= ["MedInc","HouseAge","AveRooms","AveBedrms",
                    "Population","AveOccup","Latitude","Longitude"]

    missingcolumns = [
    col for col in required_coln
    if col not in df.columns
    ]

    if missingcolumns:
        raise HTTPException(
            status_code=400,
            detail=f"These columns are missing {missingcolumns}"
        )
    if len(df)==0:
        raise HTTPException(
            status_code=400,
            detail="No data provided"
        )
    
    try:
        prediction = model.predict(df[required_coln])

        df["Predicted_columns_usd"] = prediction * 100000

        df["Predicted_columns_usd"] = df["Predicted_columns_usd"].apply(
            lambda x: f"${x:,.2f}"
        )

        output = df.to_csv(index=False)

        return StreamingResponse(
            io.StringIO(output),
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=prediction.csv"
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"prediction failed: {str(e)}"
        )