import os
import sys
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Allow importing preprocessing.py from the src folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import build_preprocessor


def load_data(data_path):
    """Load the employee attrition dataset."""
    return pd.read_csv(data_path)


def prepare_data(df):
    """Prepare features and target variable."""

    # Remove EmployeeID because it is only an identifier
    df = df.drop(columns=["EmployeeID"], errors="ignore")

    # Separate input features and target
    X = df.drop("Attrition", axis=1)

    y = df["Attrition"].map({
        "No": 0,
        "Yes": 1
    })

    return X, y


def build_model(X_train, y_train):
    """Build and train the Logistic Regression model."""

    classes = np.unique(y_train)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    class_weight_dict = dict(
        zip(classes, class_weights)
    )

    preprocessor = build_preprocessor()

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    class_weight=class_weight_dict,
                    max_iter=1000,
                    random_state=42
                )
            )
        ]
    )

    model.fit(X_train, y_train)

    return model


def train_and_save_model():
    """Train the final model and save it as a joblib file."""

    # Project root directory
    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    data_path = os.path.join(
        project_root,
        "data",
        "employee_data.csv"
    )

    model_directory = os.path.join(
        project_root,
        "models"
    )

    model_path = os.path.join(
        model_directory,
        "best_model.pkl"
    )

    # Create models directory if it does not exist
    os.makedirs(model_directory, exist_ok=True)

    # Load dataset
    df = load_data(data_path)

    # Prepare data
    X, y = prepare_data(df)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Build and train model
    model = build_model(X_train, y_train)

    # Save model
    joblib.dump(model, model_path)

    print("Model training completed successfully!")
    print(f"Model saved at: {model_path}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")


if __name__ == "__main__":
    train_and_save_model()