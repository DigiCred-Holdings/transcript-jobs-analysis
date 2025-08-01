import json
import boto3

def lambda_handler(event, context):
    # Initialize the S3 Vectors client
    s3vector = boto3.client('s3vectors', region_name='us-east-1')
    
    # Get vectors from the specified bucket and index
    response = s3vector.get_vectors(
        vectorBucketName='dev-nc-job-embeddings',
        indexName='nc-jobs',
        keys=[
            '0',
        ],
        returnData=True,
        returnMetadata=True
    )

    # Extract the vector data from the response
    vector_data = response.get('vectors', [])
    return {
        'status': 200,
        'body': vector_data
    }