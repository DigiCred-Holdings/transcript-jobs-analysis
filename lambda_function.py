import json
import boto3

def lambda_handler(event, context):

    s3vector = boto3.client('s3vector', region_name='us-east-1')
    
    response = s3vector.get_vectors(
        vectorBucketName='dev-nc-job-embeddings',
        indexName='nc-jobs',
        keys=[
            '0',
        ],
        returnData=True,
        returnMetadata=True
    )

    return {
        'status': 200,
        'body': response
    }