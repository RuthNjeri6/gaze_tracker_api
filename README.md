# Gaze Tracker API
An api that runs the gaze tracker model, Takes post request with a frame as body of request.
To test the api you should use a kivy python app found also in this repository 

## How to setup this API
1. Create a virtual environment   
   ```python3 -m venv name_of_your_environment```
2. Install all the dependencies  
   ```pip install -r requirements.txt```
3. Activate the environment  
   ```source activate name_of_your_environment/bin/activate```
4. Run the api  
   ```python3 api/app/api.py```

API should be running at [http://127.0.0.1:4000](http://127.0.0.1:4000)  
Access the routes documentation from [http://127.0.0.1:4000/docs](http://127.0.0.1:4000/docs)  

## How to setup the kivy app for testing the API
This kivy app capture a frame from the camera and send the frame as a request to the API.
The API does some prediction and returns an array of face landmarks.
To set up the app run the following commands:
1. Open a new terminal window. 
2. Activate your environment from above steps
3. Navigate to the kivy_app directory 
   ```cd kivy_app```
4. Run the app
   ```python3 main.py```
This will open a GUI window. Make sure the API is running. Happy testing!