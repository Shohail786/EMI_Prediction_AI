
import streamlit as st
import pandas as pd
import numpy as np
import mlflow
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# --- Global Variables and Setup (to be replaced with loaded objects in a real app) ---
# In a real deployed app, the scaler, label_encoder, and X_ohe_ref would be loaded from artifacts
# For this demonstration, we'll re-initialize them or assume they're available from the notebook's state

# Assume df (original preprocessed dataframe) and X (features used for training) are available
# from the notebook's previous execution state for refitting purposes.

# MLflow setup (ensure tracking URI matches where models were logged)
mlflow.set_tracking_uri("sqlite:///mlruns.db")

@st.cache_resource
def load_models():
    classification_model_uri = "models:/XGBoostClassifierModel/latest"
    regression_model_uri = "models:/RandomForestRegressorModel/latest"
    cls_model = mlflow.sklearn.load_model(classification_model_uri)
    reg_model = mlflow.sklearn.load_model(regression_model_uri)
    return cls_model, reg_model

@st.cache_resource
def get_preprocessing_objects():
    # Recreate the preprocessing objects using a dummy dataframe or load from saved artifacts
    # For this colab context, we assume 'df' and 'X' are still in memory from previous cells
    if 'df' not in st.session_state or 'X' not in st.session_state:
        st.error("Required DataFrames (df, X) not found in session state. Please run previous notebook cells.")
        st.stop()

    # Re-initialize LabelEncoder
    le = LabelEncoder()
    le.fit(st.session_state.df['emi_eligibility'])

    # Re-initialize StandardScaler
    ss = StandardScaler()
    numerical_features_to_scale = [
        'age', 'monthly_salary', 'years_of_employment', 'monthly_rent', 'family_size', 'dependents',
        'school_fees', 'college_fees', 'travel_expenses', 'groceries_utilities', 'other_monthly_expenses',
        'current_emi_amount', 'credit_score', 'bank_balance', 'emergency_fund', 'requested_amount',
        'requested_tenure', 'total_monthly_expenses', 'debt_to_income_ratio', 'expense_to_income_ratio',
        'disposable_income', 'credit_utilization_ratio', 'employment_type_score',
        'employment_stability_score', 'financial_cushion'
    ]
    ss.fit(st.session_state.df[numerical_features_to_scale])

    # Reference DataFrame for OHE columns
    X_ohe_ref = st.session_state.X.copy()
    if 'affordability_ratio' in X_ohe_ref.columns:
        X_ohe_ref = X_ohe_ref.drop(columns=['affordability_ratio'])

    return le, ss, X_ohe_ref


def preprocess_input_for_inference(input_data, scaler, X_ohe_ref):
    df_processed = input_data.copy()

    epsilon = 1e-6

    df_processed['total_monthly_expenses'] = (
        df_processed['monthly_rent'] +
        df_processed['school_fees'] +
        df_processed['college_fees'] +
        df_processed['travel_expenses'] +
        df_processed['groceries_utilities'] +
        df_processed['other_monthly_expenses']
    )
    df_processed['debt_to_income_ratio'] = df_processed['current_emi_amount'] / (df_processed['monthly_salary'] + epsilon)
    df_processed['expense_to_income_ratio'] = df_processed['total_monthly_expenses'] / (df_processed['monthly_salary'] + epsilon)
    df_processed['disposable_income'] = df_processed['monthly_salary'] - df_processed['total_monthly_expenses'] - df_processed['current_emi_amount']
    df_processed['credit_utilization_ratio'] = (df_processed['current_emi_amount'] + df_processed['requested_amount']) / (df_processed['bank_balance'] + df_processed['emergency_fund'] + epsilon)

    bins = [0, 580, 670, 740, 800, 850]
    labels = ['Very Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
    df_processed['credit_score_category'] = pd.cut(df_processed['credit_score'], bins=bins, labels=labels, right=False, ordered=True)

    employment_type_mapping = {
        'Government': 5,
        'MNC': 4,
        'Private': 3,
        'Self-employed': 2,
        'Startup': 1
    }
    df_processed['employment_type_score'] = df_processed['employment_type'].map(employment_type_mapping)
    df_processed['employment_stability_score'] = df_processed['years_of_employment'] * df_processed['employment_type_score']

    df_processed['financial_cushion'] = df_processed['bank_balance'] + df_processed['emergency_fund']

    categorical_cols_for_encoding = ['gender', 'marital_status', 'education', 'employment_type', 'company_type', 'house_type', 'existing_loans', 'emi_scenario', 'credit_score_category']
    df_processed = pd.get_dummies(df_processed, columns=categorical_cols_for_encoding, drop_first=True)

    missing_cols = set(X_ohe_ref.columns) - set(df_processed.columns)
    for c in missing_cols:
        df_processed[c] = 0
    df_processed = df_processed[X_ohe_ref.columns]

    numerical_features_to_scale = [
        'age', 'monthly_salary', 'years_of_employment', 'monthly_rent', 'family_size', 'dependents',
        'school_fees', 'college_fees', 'travel_expenses', 'groceries_utilities', 'other_monthly_expenses',
        'current_emi_amount', 'credit_score', 'bank_balance', 'emergency_fund', 'requested_amount',
        'requested_tenure', 'total_monthly_expenses', 'debt_to_income_ratio', 'expense_to_income_ratio',
        'disposable_income', 'credit_utilization_ratio', 'employment_type_score',
        'employment_stability_score', 'financial_cushion'
    ]

    df_processed[numerical_features_to_scale] = scaler.transform(df_processed[numerical_features_to_scale])

    return df_processed

# --- Streamlit UI ---
st.set_page_config(page_title="EMIPredict AI - Financial Risk Assessment", layout="wide")
st.title("🏦 EMIPredict AI - Intelligent Financial Risk Assessment")

st.markdown("This platform assesses EMI eligibility and predicts maximum affordable EMI based on your financial profile.")

# Store df and X in session state for preprocessing objects retrieval
if 'df' not in st.session_state:
    st.session_state.df = df # Assuming df is available from the colab environment
if 'X' not in st.session_state:
    st.session_state.X = X # Assuming X is available from the colab environment

# Load models and preprocessing objects
cls_model, reg_model = load_models()
label_encoder, scaler, X_ohe_ref = get_preprocessing_objects()


st.header("Applicant Financial Details")

with st.form("emi_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value= st.session_state.df['age'].max() , value=30)
        gender = st.selectbox("Gender", st.session_state.df['gender'].unique())
        marital_status = st.selectbox("Marital Status", st.session_state.df['marital_status'].unique())
        education = st.selectbox("Education Level", st.session_state.df['education'].unique())
        employment_type = st.selectbox("Employment Type", st.session_state.df['employment_type'].unique())
        years_of_employment = st.number_input("Years of Employment", min_value=0.0, max_value= st.session_state.df['years_of_employment'].max(), value=5.0)
        company_type = st.selectbox("Company Type", st.session_state.df['company_type'].unique())

    with col2:
        monthly_salary = st.number_input("Monthly Salary", min_value=0.0, max_value= st.session_state.df['monthly_salary'].max() , value=50000.0, step=1000.0)
        house_type = st.selectbox("House Type", st.session_state.df['house_type'].unique())
        monthly_rent = st.number_input("Monthly Rent", min_value=0.0, max_value= st.session_state.df['monthly_rent'].max(), value=0.0, step=100.0)
        family_size = st.number_input("Family Size", min_value=1, max_value=int(st.session_state.df['family_size'].max()), value=3)
        dependents = st.number_input("Number of Dependents", min_value=0, max_value=int(st.session_state.df['dependents'].max()), value=1)
        school_fees = st.number_input("Monthly School Fees", min_value=0.0, max_value= st.session_state.df['school_fees'].max(), value=0.0, step=100.0)
        college_fees = st.number_input("Monthly College Fees", min_value=0.0, max_value= st.session_state.df['college_fees'].max(), value=0.0, step=100.0)

    with col3:
        travel_expenses = st.number_input("Monthly Travel Expenses", min_value=0.0, max_value= st.session_state.df['travel_expenses'].max(), value=5000.0, step=100.0)
        groceries_utilities = st.number_input("Monthly Groceries/Utilities", min_value=0.0, max_value= st.session_state.df['groceries_utilities'].max(), value=10000.0, step=100.0)
        other_monthly_expenses = st.number_input("Other Monthly Expenses", min_value=0.0, max_value= st.session_state.df['other_monthly_expenses'].max(), value=2000.0, step=100.0)
        existing_loans = st.selectbox("Existing Loans", st.session_state.df['existing_loans'].unique())
        current_emi_amount = st.number_input("Current EMI Amount", min_value=0.0, max_value= st.session_state.df['current_emi_amount'].max(), value=0.0, step=100.0)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
        bank_balance = st.number_input("Bank Balance", min_value=0.0, max_value= st.session_state.df['bank_balance'].max(), value=100000.0, step=1000.0)
        emergency_fund = st.number_input("Emergency Fund", min_value=0.0, max_value= st.session_state.df['emergency_fund'].max(), value=50000.0, step=1000.0)
        requested_amount = st.number_input("Requested Loan Amount", min_value= st.session_state.df['requested_amount'].min(), max_value= st.session_state.df['requested_amount'].max(), value=50000.0, step=1000.0)
        requested_tenure = st.number_input("Requested Tenure (Years)", min_value=1, max_value=int(st.session_state.df['requested_tenure'].max()), value=5)
        emi_scenario = st.selectbox("EMI Scenario", st.session_state.df['emi_scenario'].unique())

    submitted = st.form_submit_button("Get EMI Assessment")

    if submitted:
        input_df = pd.DataFrame([{
            'age': age,
            'gender': gender,
            'marital_status': marital_status,
            'education': education,
            'monthly_salary': monthly_salary,
            'employment_type': employment_type,
            'years_of_employment': years_of_employment,
            'company_type': company_type,
            'house_type': house_type,
            'monthly_rent': monthly_rent,
            'family_size': family_size,
            'dependents': dependents,
            'school_fees': school_fees,
            'college_fees': college_fees,
            'travel_expenses': travel_expenses,
            'groceries_utilities': groceries_utilities,
            'other_monthly_expenses': other_monthly_expenses,
            'existing_loans': existing_loans,
            'current_emi_amount': current_emi_amount,
            'credit_score': credit_score,
            'bank_balance': bank_balance,
            'emergency_fund': emergency_fund,
            'emi_scenario': emi_scenario,
            'requested_amount': requested_amount,
            'requested_tenure': requested_tenure,
            # 'emi_eligibility': 'N/A', # Target, not input
            # 'max_monthly_emi': 0.0 # Target, not input
        }])

        try:
            processed_input = preprocess_input_for_inference(input_df, scaler, X_ohe_ref)

            # Make predictions
            emi_eligibility_pred_encoded = cls_model.predict(processed_input)
            emi_eligibility_pred = label_encoder.inverse_transform(emi_eligibility_pred_encoded)
            max_monthly_emi_pred = reg_model.predict(processed_input)

            st.subheader("Prediction Results")
            st.success(f"EMI Eligibility Status: **{emi_eligibility_pred[0]}**")
            st.info(f"Estimated Maximum Monthly EMI: **₹{max_monthly_emi_pred[0]:.2f}**")

            if emi_eligibility_pred[0] == 'Eligible':
                st.balloons()
                st.write("Congratulations! You appear to be eligible for the EMI based on your profile.")
            elif emi_eligibility_pred[0] == 'High_Risk':
                st.warning("Your application is flagged as High Risk. Further review may be required.")
            else:
                st.error("Unfortunately, you appear to be Not Eligible for the EMI at this time.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

st.markdown("---")
st.markdown("Disclaimer: This is an AI-powered assessment and should not be considered final financial advice.")
