from lambda_function import lambda_handler


def test_lambda_handler():
    event = {}
    context = {}
    response = lambda_handler(event, context)
    
    assert response['status'] == 200
    assert 'body' in response
    assert response['body'] == {"Hello world"}
