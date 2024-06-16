import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from xgboost import XGBRegressor
import mlflow
import mlflow.sklearn

# Load the dataset
print("Loading dataset...")
df = pd.read_csv("goat_dataset.csv")
print("Dataset loaded successfully!")

# Drop unnecessary columns
print("Dropping unnecessary columns...")
df.drop(["owner_name", "adhaar_number", "goat_id"], axis=1, inplace=True)

# Map categorical data to numerical values
print("Mapping categorical data to numerical values...")
pregnancy_dict = {"no": 0, "yes": 1}
df['pregnancy'] = df['pregnancy'].map(pregnancy_dict)
behavior_dict = {"docile": 0, "normal": 1, "aggressive": 2}
df['behavior'] = df['behavior'].map(behavior_dict)
gender_dict = {"female": 0, "male": 1}
df['gender'] = df['gender'].map(gender_dict)

# Define features and target variable
print("Defining features and target variable...")
x = df.drop(["meat_quality_of_the_goat", "milk_quality_of_the_goat"], axis=1)
y = df[["meat_quality_of_the_goat"]]

# Split the dataset into training and testing sets
print("Splitting the dataset into training and testing sets...")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(f"Training set size: {x_train.shape[0]}")
print(f"Test set size: {x_test.shape[0]}")

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

import dagshub
dagshub.init(repo_owner='codeInterpreter05', repo_name='my-first-repo', mlflow=True)

# Start MLflow experiment
print("Starting MLflow experiment...")
# mlflow.set_tracking_url('')
mlflow.set_experiment("Model Tracking Experiment")

for model_name, model in models.items():
    print(f"Training model: {model_name}...")
    
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)
        
        # Train model
        model.fit(x_train, y_train)
        print(f"Model {model_name} trained successfully!")

        # Make predictions
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)

        # Evaluate Train and Test dataset
        model_train_mae, model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)
        model_test_mae, model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)

        print(f"Evaluating model: {model_name}...")
        model_list.append(model_name)

        print('Model performance for Training set')
        print(f"- Root Mean Squared Error: {model_train_rmse:.4f}")
        print(f"- Mean Absolute Error: {model_train_mae:.4f}")
        print(f"- R2 Score: {model_train_r2:.4f}")
        print('----------------------------------')

        print('Model performance for Test set')
        print(f"- Root Mean Squared Error: {model_test_rmse:.4f}")
        print(f"- Mean Absolute Error: {model_test_mae:.4f}")
        print(f"- R2 Score: {model_test_r2:.4f}")
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
        print(f"Model {model_name} logged to MLflow successfully!\n")

print("All models have been trained and logged successfully!")
