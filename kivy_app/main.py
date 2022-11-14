from multiprocessing.pool import ThreadPool
from kivy.app import App
from kivy.uix.image import Image
from kivy.graphics import Ellipse, Color
from kivy.network.urlrequest import UrlRequest
import cv2
import json
import requests
from dotenv import load_dotenv
import os

class Touch(Image):
    # only send frame on touch/click event
    # add a button to save data collected
    """on touch down, get cap and get frame"""
    def on_touch_down(self, touch):
        global video_frame, collected_x, collected_y
        x = touch.x
        y = touch.y
        if len(video_frame) != 0:
            prediction = self.sendFrame(video_frame)
            print('Prediction...', prediction)
            if prediction is not None and len(prediction) == 28:
                with self.canvas:
                    Color(1,0,0)
                    d = 20
                    Ellipse(pos=(x -d/2, y - d/2), size=(d, d))
                collected_x.append(prediction)
                collected_y.append((x,y))
            video_frame = []

    def sendFrame(self, img):
        url = os.environ.get('URL') + '/predict'
        payload = json.dumps({
                    "frame": img,
        })
        headers = {
            'Content-Type': 'application/json'
        }
        req = UrlRequest(url=url, req_headers=headers, req_body=payload)
        req.wait()
        # response = UrlRequest(url=url, req_headers=headers, req_body=payload, ca_file=cfi.where(), verify=True)
        data = req.result
        return data['prediction']

class MainApp(App):
    def build(self):
        return Touch()
    def on_start(self):
        self.save_current = False
        self.pool = ThreadPool(processes=1)
        async_start = self.pool.apply_async(self.getFrame)
    
    def getFrame(self):
        global video_frame
        cap = cv2.VideoCapture(0)
        while True:
            ret, image = cap.read()
            if not ret:
                print("Frame cannot be read. Exiting...")
                video_frame = []
                break
            else:
                video_frame = image.tolist()
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break 
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def on_stop(self):
        status = self.saveData()
        if status:
            # forcefully close all worker threads
            print('data send successfully')
            self.pool.terminate()

    def saveData(self):
        url = os.environ.get('URL') + '/save'
        payload = json.dumps({
                    "landmarks": collected_x,
                    "labels": collected_y
        })
        headers = {
            'Content-Type': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        data = json.loads(response.text)
        status = data['status']
        return status

if __name__ == '__main__':
    load_dotenv()
    video_frame = []
    collected_x = []
    collected_y = []
    MainApp().run()
