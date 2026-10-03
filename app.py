import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="👨‍💼",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #f5f7fb;
}

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Main title */
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    color: #1e3a8a;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #64748b;
    margin-bottom: 30px;
}

/* Section headings */
.section-heading {
    font-size: 25px;
    font-weight: 700;
    color: #1e293b;
    margin-top: 20px;
    margin-bottom: 15px;
}

/* Information box */
.info-box {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #e2e8f0;
    margin-bottom: 25px;
}

/* Button */
.stButton > button {
    width: 100%;
    height: 52px;
    border-radius: 12px;
    font-size: 18px;
    font-weight: 600;
    background-color: #2563eb;
    color: white;
    border: none;
}

.stButton > button:hover {
    background-color: #1d4ed8;
    color: white;
}

/* Result */
.result-box {
    background-color: white;
    padding: 30px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid #e2e8f0;
    margin-top: 25px;
}

.result-title {
    font-size: 28px;
    font-weight: 700;
}

.probability {
    font-size: 42px;
    font-weight: 800;
    margin-top: 10px;
}


</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

try:

    model = joblib.load("models/best_model.pkl")

except FileNotFoundError:

    st.error(
        "❌ Model file not found. "
        "Please make sure models/best_model.pkl exists."
    )

    st.stop()

except Exception as error:

    st.error(f"❌ Error loading model: {error}")

    st.stop()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">👨‍💼 Employee Attrition Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent Workforce Risk Analysis using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# ABOUT SECTION
# =========================================================

st.info(
    "📌 About the System\n\n"
    "This system analyzes employee information and predicts "
    "the likelihood of employee attrition using a trained "
    "Machine Learning model."
)


# =========================================================
# EMPLOYEE INFORMATION
# =========================================================

st.markdown(
    '<div class="section-heading">📋 Employee Information</div>',
    unsafe_allow_html=True
)


col1, col2 = st.columns(2)


# ---------------------------------------------------------
# LEFT COLUMN
# ---------------------------------------------------------

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=70,
        value=30,
        step=1
    )

    department = st.selectbox(
        "Department",
        [
            "Sales",
            "Engineering",
            "HR",
            "Marketing",
            "Finance"
        ]
    )

    monthly_income = st.number_input(
        "Monthly Income (₹)",
        min_value=0.0,
        value=30000.0,
        step=1000.0
    )

    years_at_company = st.number_input(
        "Years at Company",
        min_value=0,
        max_value=50,
        value=5,
        step=1
    )

    distance_from_home = st.number_input(
        "Distance From Home (km)",
        min_value=0.0,
        value=10.0,
        step=1.0
    )


# ---------------------------------------------------------
# RIGHT COLUMN
# ---------------------------------------------------------

with col2:

    job_satisfaction = st.slider(
        "Job Satisfaction",
        min_value=1,
        max_value=5,
        value=3
    )

    work_life_balance = st.slider(
        "Work-Life Balance",
        min_value=1,
        max_value=5,
        value=3
    )

    performance_rating = st.slider(
        "Performance Rating",
        min_value=1,
        max_value=5,
        value=3
    )

    overtime = st.selectbox(
        "OverTime",
        ["Yes", "No"]
    )


# =========================================================
# VALIDATION
# =========================================================

def validate_inputs():

    if age < 18 or age > 70:
        return "Age must be between 18 and 70."

    if monthly_income < 0:
        return "Monthly income cannot be negative."

    if years_at_company < 0:
        return "Years at company cannot be negative."

    if years_at_company > age:
        return (
            "Years at company cannot be greater "
            "than the employee's age."
        )

    if job_satisfaction < 1 or job_satisfaction > 5:
        return "Job satisfaction must be between 1 and 5."

    if work_life_balance < 1 or work_life_balance > 5:
        return "Work-life balance must be between 1 and 5."

    if performance_rating < 1 or performance_rating > 5:
        return "Performance rating must be between 1 and 5."

    if distance_from_home < 0:
        return "Distance from home cannot be negative."

    return None


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

if st.button(
    "🔍 Predict Employee Attrition",
    use_container_width=True
):

    validation_error = validate_inputs()

    if validation_error:

        st.error(f"⚠️ {validation_error}")

    else:

        try:

            input_data = pd.DataFrame({
                "Age": [age],
                "Department": [department],
                "MonthlyIncome": [monthly_income],
                "YearsAtCompany": [years_at_company],
                "JobSatisfaction": [job_satisfaction],
                "WorkLifeBalance": [work_life_balance],
                "PerformanceRating": [performance_rating],
                "OverTime": [overtime],
                "DistanceFromHome": [distance_from_home]
            })

            # Check missing values
            if input_data.isnull().any().any():

                st.error(
                    "❌ Missing or invalid input data detected."
                )

                st.stop()

            # Prediction
            prediction = model.predict(input_data)[0]

            probability = model.predict_proba(
                input_data
            )[0][1]

            probability_percentage = probability * 100


            # =================================================
            # RESULT
            # =================================================

            st.markdown(
                '<div class="section-heading">📊 Prediction Result</div>',
                unsafe_allow_html=True
            )


            if prediction == 1:

                st.error(
                    "⚠️ Employee may leave"
                )

            else:

                st.success(
                    "✅ Employee likely to stay"
                )


            st.metric(
                "Attrition Probability",
                f"{probability_percentage:.2f}%"
            )


            # =================================================
            # EMPLOYEE SUMMARY
            # =================================================

            st.markdown(
                '<div class="section-heading">👤 Employee Summary</div>',
                unsafe_allow_html=True
            )

            summary1, summary2, summary3, summary4 = st.columns(4)

            with summary1:

                st.metric(
                    "Age",
                    age
                )

            with summary2:

                st.metric(
                    "Department",
                    department
                )

            with summary3:

                st.metric(
                    "Monthly Income",
                    f"₹{monthly_income:,.0f}"
                )

            with summary4:

                st.metric(
                    "OverTime",
                    overtime
                )


            # =================================================
            # COMPLETE DETAILS
            # =================================================

            with st.expander(
                "📄 View Complete Employee Details"
            ):

                st.dataframe(
                    input_data,
                    use_container_width=True,
                    hide_index=True
                )


        except Exception as error:

            st.error(
                f"❌ Prediction error: {error}"
            )




