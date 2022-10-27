import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from predict import update
from save import save_data


class Predictor(BaseModel):
    frame: list
class Data(BaseModel):
    landmarks: list
    labels: list 

app = FastAPI()

@app.get('/')
def index():
    return {'message': 'This is the homepage of the API '}

@app.post('/predict')
def predict(data: Predictor):
    data = data.dict()
    frame = np.asarray(data['frame'], dtype="uint8")
    prediction = update(frame)
    print(prediction)
    if prediction is not None:
        prediction = prediction.tolist()
    return {'prediction' : prediction}

@app.post('/save')
def save(data: Data):
    data = data.dict()
    landmarks = data['landmarks']
    labels = data['labels']
    status = save_data(landmarks, labels)
    return {'status': status}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=4000, debug=True)