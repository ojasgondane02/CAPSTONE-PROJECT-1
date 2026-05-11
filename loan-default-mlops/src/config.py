# Configuration settings for the ML project

# Random state for reproducibility
RANDOM_STATE = 42

# Train-test split size
TEST_SIZE = 0.2

# Models to train
MODELS = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Naive Bayes",
    "KNN"
]

# Hyperparameter grids (kept small for simplicity)
PARAM_GRIDS = {
    "Random Forest": {
        "n_estimators": [50, 100],
        "max_depth": [None, 5, 10]
    },
    "Decision Tree": {
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5]
    },
    "KNN": {
        "n_neighbors": [3, 5, 7]
    }
}
