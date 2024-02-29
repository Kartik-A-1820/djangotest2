from threading import Thread
import time

# Import your functions and dependencies
from .models import Results
import boto3
from django.db import IntegrityError, transaction
import pickle

# Maximum number of retries
MAX_RETRIES = 3
# Delay between retries in seconds
RETRY_DELAY = 0.5  # Half a second delay

def get_latest_user():
    try:
        with open('djangotestapp/user.pkl', 'rb') as file:
            user = pickle.load(file).split('@')[0]
    except:
        user = None
    return user

def upload_to_s3(body, bucket, key):
    AWS_ACCESS_KEY_ID = 'AKIA2QMGQOEAXYO3BIYV'
    AWS_SECRET_ACCESS_KEY = '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'
    s3 = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    s3.put_object(Body=body, Bucket=bucket, Key=key)

def create_result_with_retry(fishname, input_image_url, result_image_url, total, good, bad, reasons):
    retries = 0
    while retries < MAX_RETRIES:
        try:
            last_result = Results.objects.order_by('-serial_number').first()
            last_serial_number = last_result.serial_number
            serial_number = last_serial_number + 1
            Results.objects.create(
                    serial_number=serial_number,
                    species_name=fishname,
                    input_image_url=input_image_url,
                    output_image_url=result_image_url,
                    total_count=total,
                    good_count=good,
                    bad_count=bad,
                    prediction_reason=reasons,
                    user_name=get_latest_user(),
                )
                # If the creation is successful, exit the loop
            break
        except IntegrityError:
            # If IntegrityError occurs, increment the retry counter
            retries += 1
            if retries < MAX_RETRIES:
                # If there are more retries left, wait before retrying
                time.sleep(RETRY_DELAY)
            else:
                # If no more retries left, log the error or handle as needed
                print("Error: Max retries exceeded. Unable to create result.")

# Function to run upload_to_s3 asynchronously
def run_upload_to_s3(body, bucket, key):
    thread = Thread(target=upload_to_s3, args=(body, bucket, key))
    thread.start()

# Function to run create_result_with_retry asynchronously
def run_create_result_with_retry(fishname, input_image_url, result_image_url, total, good, bad, reasons):
    thread = Thread(target=create_result_with_retry, args=(fishname, input_image_url, result_image_url, total, good, bad, reasons))
    thread.start()

