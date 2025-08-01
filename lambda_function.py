import json
import boto3

def lambda_handler(event, context):
    return {
        'status': 200,
        'body': "Hello world"
    }