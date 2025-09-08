import os
import numpy as np
import json
import mlflow   
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI

class FetalHealthData(BaseModel):
    accelerations: float
    fetal_movement: float
    uterine_contractions: float
    severe_decelerations: float


app = FastAPI(title="Fetal Health API", 
              openapi_tags=[
                  {
                      "name": "Health",
                      "description": "Get api health"
                      },
                  {
                        "name": "Prediction",
                        "description": "Model prediction"
                  }
              ])

def load_model():
    print("Loading model...")
    MLFLOW_TRACKING_URI = 'https://dagshub.com/renansantosmendes/mlops-ead.mlflow'
    MLFLOW_TRACKING_USERNAME = 'renansantosmendes'
    MLFLOW_TRACKING_PASSWORD = '6d730ef4a90b1caf28fbb01e5748f0874fda6077'
    os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD
    print("set environment variables")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print("craating client")
    client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    print("getting registered model")
    registered_model = client.get_registered_model('fetal_health')
    print("read model")
    run_id = registered_model.latest_versions[-1].run_id
    logged_model = f'runs:/{run_id}/model'
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    print(loaded_model)
    print("model loaded")
    
    return loaded_model

@app.on_event("startup")
def startup_event():
    global loaded_model 
    loaded_model = load_model()

@app.get("/", tags=["Health"])
def api_health():
    return {"message": "Hello World"}

@app.post("/predict", tags=["Prediction"])
def predict(request: FetalHealthData):
    global loaded_model


    received_data = np.array([
        request.accelerations,
        request.fetal_movement,
        request.uterine_contractions,
        request.severe_decelerations,
    ]).reshape(1, -1)
    prediction = loaded_model.predict(received_data)
    return {"predict": str(np.argmax(prediction[0]))}