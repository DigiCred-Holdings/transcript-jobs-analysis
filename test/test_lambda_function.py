import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import lambda_function


def test_lambda_handler():
    event = {}
    context = {}
    response = lambda_function.lambda_handler(event, context)
    
    assert response['status'] == 200
    assert 'body' in response
    assert response['body'] == {"Hello world"}
