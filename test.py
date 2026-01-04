import mlflow
print('Printing Tracking URI sqlite')
print(mlflow.get_tracking_uri())
print('\n')

mlflow.set_tracking_uri('http://127.0.0.1:5000')
print('Printing New Tracking URI')
print(mlflow.get_tracking_uri())