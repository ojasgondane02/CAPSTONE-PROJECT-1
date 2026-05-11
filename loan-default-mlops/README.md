# Loan Default Prediction - Capstone Project

Welcome to the Loan Default Prediction Machine Learning & MLOps project. This project is built to demonstrate an end-to-end Machine Learning pipeline starting from Data Analysis in Jupyter Notebook to API deployment using FastAPI and experiment tracking via MLflow.

## Project Overview

This system predicts whether a loan applicant is likely to default on their loan based on various input features.

**Key Features:**
- **Data Preprocessing & EDA:** Clean, analyze, and visualize data.
- **Model Training:** Train multiple models (Logistic Regression, Decision Trees, Random Forest, etc.).
- **Imbalance Handling:** Uses SMOTE to handle minority classes.
- **MLOps Integration:** MLflow tracks models, parameters, and metrics.
- **REST API:** FastAPI application serves the model for real-time predictions.
- **CI/CD:** Basic GitHub Actions workflow for continuous integration.

## Folder Structure

```
loan-default-mlops/
│
├── data/                       # Store raw and processed data (e.g., Loan_Default.csv)
├── notebooks/                  # Jupyter notebooks for EDA and training
│   └── loan_default_training.ipynb
├── src/                        # Core Python scripts
│   ├── preprocessing.py        # Data cleaning, scaling, encoding
│   ├── train_utils.py          # Model training and hyperparameter tuning
│   ├── evaluate.py             # Metrics calculation and plotting
│   └── config.py               # Settings and constants
├── models/                     # Trained models saved as .pkl
├── api/                        # FastAPI REST API code
│   ├── app.py                  # API endpoints
│   └── schema.py               # Pydantic validation schemas
├── mlflow/                     # MLflow tracking utilities
│   └── mlflow_tracking.py
├── .github/workflows/          # CI/CD pipelines
│   └── ci.yml
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Getting Started

### 1. Installation

Clone this repository and set up your virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Jupyter Notebook

Navigate to the `notebooks` directory and start Jupyter:

```bash
jupyter notebook notebooks/loan_default_training.ipynb
```

Follow the cell-by-cell instructions in the notebook to clean data, train models, and save the best model to the `models/` directory.

### 3. Running the FastAPI App

Once you have generated the `best_model.pkl` file, you can start the API:

```bash
uvicorn api.app:app --reload
```

The API will be available at `http://127.0.0.1:8000`.
You can access the Swagger UI documentation at `http://127.0.0.1:8000/docs`.

### 4. Sample API Request

You can test the prediction endpoint (`/predict`) using tools like Postman or cURL:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "age": 35,
  "income": 60000.0,
  "loan_amount": 15000.0,
  "credit_score": 720.0,
  "employment_years": 5.0,
  "gender": 1
}'
```

### 5. Tracking with MLflow

To view the tracked MLflow experiments, run the following command in your terminal:

```bash
mlflow ui
```

Then open `http://127.0.0.1:5000` in your web browser.

## Future Improvements

- Deploy the FastAPI application to Azure App Services or AWS EC2.
- Add advanced CI/CD pipelines for automatic model deployment.
- Implement Docker for containerizing the application.
- Use a robust relational database like PostgreSQL for storing predictions.
