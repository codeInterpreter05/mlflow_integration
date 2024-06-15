import numpy as np
import pandas as pd

df = pd.read_csv("goat_dataset.csv")

df.drop(["owner_name" , "adhaar_number" , "goat_id"] , axis = 1 , inplace = True)

pregnancy_dict = {
    "no" : 0,
    "yes" : 1
}
df['pregnancy'] = df['pregnancy'].map(pregnancy_dict)
behavior_dict = {
    "docile" : 0,
    "normal" : 1,
    "aggressive" : 2
}
df['behavior'] = df['behavior'].map(behavior_dict)
gender_dict = {
    "female" : 0,
    "male" : 1
}
df['gender'] = df['gender'].map(gender_dict)

x = df.drop(["meat_quality_of_the_goat" , "milk_quality_of_the_goat"] , axis = 1 )
y = df[["meat_quality_of_the_goat"]]

from sklearn.model_selection import train_test_split

# Assuming X and y are your features and labels respectively
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(f"Training set size: {x_train.shape[0]}")
print(f"Test set size: {x_test.shape[0]}")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    return mae, rmse, r2_square


X_train = x_train
X_test = x_test

import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "K-Neighbors Regressor": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "XGBRegressor": XGBRegressor(), 
    "AdaBoost Regressor": AdaBoostRegressor()
}

model_list = []
r2_list = []

# Function to evaluate the model
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

# Start MLflow experiment
mlflow.set_experiment("Model Tracking Experiment")

for i in range(len(list(models))):
    model_name = list(models.keys())[i]
    model = list(models.values())[i]

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_version", i+1)
        
        model.fit(X_train, y_train)  # Train model

        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        print(y_train_pred)

        # Evaluate Train and Test dataset
        model_train_mae, model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)
        model_test_mae, model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)

        print(model_name)
        model_list.append(model_name)

        print('Model performance for Training set')
        print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
        print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
        print("- R2 Score: {:.4f}".format(model_train_r2))

        print('----------------------------------')

        print('Model performance for Test set')
        print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
        print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
        print("- R2 Score: {:.4f}".format(model_test_r2))
        r2_list.append(model_test_r2)

        print('='*35)
        print('\n')

        # Log metrics to MLflow
        mlflow.log_metric("train_rmse", model_train_rmse)
        mlflow.log_metric("train_mae", model_train_mae)
        mlflow.log_metric("train_r2", model_train_r2)
        mlflow.log_metric("test_rmse", model_test_rmse)
        mlflow.log_metric("test_mae", model_test_mae)
        mlflow.log_metric("test_r2", model_test_r2)

        # Log the model
        mlflow.sklearn.log_model(model, model_name)