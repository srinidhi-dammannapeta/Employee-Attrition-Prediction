import os
import joblib


VALID_DEPARTMENTS = [
    "Sales",
    "Engineering",
    "HR",
    "Marketing",
    "Finance"
]

VALID_OVERTIME_VALUES = [
    "Yes",
    "No"
]


def validate_employee_data(
    age,
    department,
    monthly_income,
    years_at_company,
    job_satisfaction,
    work_life_balance,
    performance_rating,
    overtime,
    distance_from_home
):
    """
    Validate employee input values.

    Returns:
        None if all inputs are valid.
        Error message if an input is invalid.
    """

    if age < 18 or age > 70:
        return "Age must be between 18 and 70."

    if department not in VALID_DEPARTMENTS:
        return "Invalid department."

    if monthly_income < 0:
        return "Monthly income cannot be negative."

    if years_at_company < 0:
        return "Years at company cannot be negative."

    if years_at_company > age:
        return "Years at company cannot be greater than employee age."

    if job_satisfaction < 1 or job_satisfaction > 5:
        return "Job satisfaction must be between 1 and 5."

    if work_life_balance < 1 or work_life_balance > 5:
        return "Work-life balance must be between 1 and 5."

    if performance_rating < 1 or performance_rating > 5:
        return "Performance rating must be between 1 and 5."

    if overtime not in VALID_OVERTIME_VALUES:
        return "Invalid overtime value."

    if distance_from_home < 0:
        return "Distance from home cannot be negative."

    return None


def load_trained_model():
    """
    Load the saved employee attrition model.
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
            "best_model.pkl was not found in the models folder."
        )

    try:
        model = joblib.load(model_path)
        return model

    except Exception as error:
        raise RuntimeError(
            f"Error loading the trained model: {error}"
        )