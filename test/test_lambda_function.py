import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import lambda_function
from unittest.mock import patch, MagicMock

def test_lambda_handler():
    event = {}
    context = {}

    mock_response = {'vectors': ['mocked_vector']}
    with patch('boto3.client') as mock_client:
        mock_s3vector = MagicMock()
        mock_s3vector.get_vectors.return_value = mock_response
        mock_client.return_value = mock_s3vector

        response = lambda_function.lambda_handler(event, context)

    assert response['status'] == 200
    assert 'body' in response
    assert response['body'] == ['mocked_vector']
