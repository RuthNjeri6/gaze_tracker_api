from fastapi import FastAPI
from pydantic import BaseModel

class Predictor(BaseModel):
    frame: list

app = FastAPI()

@app.get("/")
def index():
    return {"message":"This is the homepage of the API"}

@app.post("/predict")
def predict(data: Predictor):
    return {"message": data}