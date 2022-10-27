from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from predict import update

class Predictor(BaseModel):
    frame: list

app = FastAPI()

@app.get("/")
def index():
    return {"message":"This is the homepage of the API"}

@app.post("/predict")
def predict(data: Predictor):
    data = data.dict()
    frame = np.asarray(data['frame'], dtype="uint8")
    prediction = update(frame)
    return {"message": prediction}