import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_iris()
x, y = data.data, data.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Start an MLflow run
mlflow.start_run()

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

mlflow.log_param("model_iter", model.max_iter)
mlflow.log_metric("accuracy", accuracy)
mlflow.sklearn.log_model(model, "model")
# End the MLflow run
mlflow.end_run()
