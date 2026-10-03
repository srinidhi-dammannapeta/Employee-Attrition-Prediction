import os
import joblib
import pandas as pd


def load_model():
    """
    Load the trained employee attrition model.
    """

    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    model_path = os.path.join(
        project_root,
        "models",
        "best_model.pkl"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Model file not found. Please train the model first."
        )

    return joblib.load(model_path)


def predict_attrition(employee_data):
    """
    Predict employee attrition and return
    prediction and probability.
    """

    model = load_model()

    # Convert input into DataFrame
    input_data = pd.DataFrame([employee_data])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Get probability of attrition
    probability = model.predict_proba(input_data)[0][1]

    result = "Yes" if prediction == 1 else "No"

    return result, probability


if __name__ == "__main__":

    sample_employee = {
        "Age": 45,
        "Department": "Engineering",
        "MonthlyIncome": 60000,
        "YearsAtCompany": 15,
        "JobSatisfaction": 5,
        "WorkLifeBalance": 4,
        "PerformanceRating": 4,
        "OverTime": "No",
        "DistanceFromHome": 5
    }

    prediction, probability = predict_attrition(
        sample_employee
    )

    print("Employee Attrition Prediction")
    print("-----------------------------")
    print(f"Prediction: {prediction}")
    print(f"Attrition Probability: {probability * 100:.2f}%")