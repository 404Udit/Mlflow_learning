import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load wine dataset
wine=load_wine()
X=wine.data
Y=wine.target

# train-test-split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,random_state=42)

# Define the params for RF model
max_depth=10
n_estimators=10

with mlflow.start_run():
    rf=RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators,random_state=42)
    rf.fit(X_train,Y_train)

    y_pred=rf.predict(X_test)
    accuracy=accuracy_score(Y_test,y_pred)

    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_param('max_depth',max_depth)
    mlflow.log_param('n_estimators',n_estimators)

    # Creating a confusion matrix plot
    cm=confusion_matrix(Y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=wine.target_names,yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('predicted')
    plt.title('Confusion Matrix')

    # Save figure
    plt.savefig('Confusion_matrix.png')

    # log artifacts using mlflow
    mlflow.log_artifact('Confusion_matrix.png')
    mlflow.log_artifact(__file__)

    print(accuracy)
