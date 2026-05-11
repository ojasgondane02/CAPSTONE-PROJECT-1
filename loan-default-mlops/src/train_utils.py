from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import joblib

def get_models(random_state=42):
    """
    Returns a dictionary of uninitialized models.
    """
    return {
        "Logistic Regression": LogisticRegression(random_state=random_state, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(random_state=random_state),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier()
    }

def train_model(model, X_train, y_train):
    """
    Trains a single model.
    """
    model.fit(X_train, y_train)
    return model

def tune_hyperparameters(model, param_grid, X_train, y_train, cv=3):
    """
    Performs hyperparameter tuning using GridSearchCV.
    """
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def save_model(model, filepath):
    """
    Saves the trained model to a file using joblib.
    """
    joblib.dump(model, filepath)
    print(f"Model saved successfully at: {filepath}")
