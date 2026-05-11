import mlflow
import mlflow.sklearn
import os

def track_experiment(experiment_name, run_name, model, params, metrics, X_train):
    """
    Tracks an ML experiment using MLflow.
    """
    # Create or set the experiment
    mlflow.set_experiment(experiment_name)
    
    # Start an MLflow run
    with mlflow.start_run(run_name=run_name):
        # Log hyperparameters
        mlflow.log_params(params)
        
        # Log evaluation metrics
        mlflow.log_metrics(metrics)
        
        # Infer the model signature (input and output schema)
        from mlflow.models.signature import infer_signature
        
        # For a simple dummy prediction to infer signature
        sample_input = X_train[:5]
        sample_output = model.predict(sample_input)
        signature = infer_signature(sample_input, sample_output)
        
        # Log the trained model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name=f"{run_name}_Model"
        )
        
        print(f"Successfully logged run: {run_name} in experiment: {experiment_name}")

if __name__ == "__main__":
    print("This file contains helper functions for MLflow tracking. Import it to use in your scripts or notebooks.")
