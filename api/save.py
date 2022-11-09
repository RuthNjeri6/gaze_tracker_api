from dotenv import load_dotenv
import boto3
import os
import numpy as np
from datetime import datetime
from multiprocessing.pool import ThreadPool

load_dotenv()
region = os.environ.get('AWS_REGION')
id = os.environ.get('AWS_ACCESS_KEY_ID')
key = os.environ.get('AWS_SECRET_ACCESS_KEY')
bucket = os.environ.get('BUCKET_NAME')

s3_client = boto3.client(
        's3',
        aws_access_key_id = id,
        aws_secret_access_key = key,
        region_name = region
)

def upload_to_s3(local_file, remote_file):
    """ function that uploads file to s3 bucket """
    try:
        s3_client.upload_file(local_file, bucket, remote_file)
    except Exception as err:
        print(err)
    else:
        os.remove(local_file)

def save_data(landmarks, labels):
    save_time = datetime.now().strftime("%H_%M_%S")
    landmarks_file = save_time + '_landmarks.txt'
    labels_file = save_time + '_labels.txt'
    try:
        np.savetxt('./saved_data/' + landmarks_file, landmarks, fmt='%i', delimiter =',')
        np.savetxt('./saved_data/' + labels_file, labels, fmt='%i', delimiter =',')
    except Exception as err:
        print(err)
    else:
        pool = ThreadPool(processes=1)
        async_landmarks = pool.apply_async(upload_to_s3, args=(landmarks_file, bucket, 'gaze_tracker/' + landmarks_file))
        async_labels = pool.apply_async(upload_to_s3, args=(labels_file, bucket, 'gaze_tracker/' + labels_file))
        return async_landmarks.ready
        # if async_landmarks.successful:
        #     os.remove('./save_data/' + landmarks_file)
        # if async_labels.successful:
        #     os.remove('./saved_data/' + labels_file)

    