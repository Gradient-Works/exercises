import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os
import boto3
import logging

load_dotenv()

def load_data(file_name):
    """
    Load the data from the CSV file.

    Returns:
        pandas.DataFrame: The DataFrame containing the loaded data.
    """
    return pd.read_csv(file_name)

def create_client():
    """
    Creates and returns an instance of the OpenAI client.

    Returns:
        client: An instance of the OpenAI client.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return client

def load_data_from_s3(bucket_name, file_key):
    """
    Load data from an S3 bucket.

    Args:
        bucket_name (str): The name of the S3 bucket.
        file_key (str): The key (path) of the file in the S3 bucket.

    Returns:
        file: The file object loaded from S3.
    """
    try:
        s3 = boto3.client('s3', aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"), aws_secret_access_key=os.getenv("AWS_SECRET"))
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        return response['Body']

    except Exception as e:
        logging.error(f"Failed to load data from S3 due to: {e}")
        return None