from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def get_feature_lists():
    """
    Return the numerical and categorical feature names
    used in the employee attrition model.
    """

    numerical_features = [
        "Age",
        "MonthlyIncome",
        "YearsAtCompany",
        "JobSatisfaction",
        "WorkLifeBalance",
        "PerformanceRating",
        "DistanceFromHome"
    ]

    categorical_features = [
        "Department",
        "OverTime"
    ]

    return numerical_features, categorical_features


def build_preprocessor():
    """
    Build and return the preprocessing pipeline.

    Numerical features are standardized using StandardScaler.
    Categorical features are encoded using OneHotEncoder.
    """

    numerical_features, categorical_features = get_feature_lists()

    numerical_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor