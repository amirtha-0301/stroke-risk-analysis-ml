
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Stroke Risk Intelligence",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.main-title {
    font-size: 42px;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 17px;
    color: #64748b;
    margin-bottom: 25px;
}

.kpi-card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #e2e8f0;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.05);
}

.kpi-title {
    color: #64748b;
    font-size: 14px;
}

.kpi-value {
    color: #0f172a;
    font-size: 28px;
    font-weight: 700;
}

.section-title {
    font-size: 25px;
    font-weight: 650;
    color: #0f172a;
    margin-top: 20px;
}

.warning-box {
    background-color: #fff7ed;
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid #f97316;
    margin-bottom: 20px;
}

.info-box {
    background-color: #eff6ff;
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid #3b82f6;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv("stroke_risk_cleaned.csv")

    return df


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load("stroke_risk_model.pkl")

    features = joblib.load("stroke_risk_features.pkl")

    return model, features


# ============================================================
# LOAD PROJECT FILES
# ============================================================

try:

    df = load_data()

    model, feature_columns = load_model()

except Exception as e:

    st.error("Unable to load the project files.")

    st.write(
        "Make sure the following files are present in the GitHub repository:"
    )

    st.code("""
stroke_risk_cleaned.csv
stroke_risk_model.pkl
stroke_risk_features.pkl
""")

    st.error(str(e))

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🩺 Stroke Risk Intelligence")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "Dashboard",
        "Risk Factor Analysis",
        "Patient Risk Profile",
        "Stroke Risk Prediction"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Academic machine-learning project for stroke risk analysis "
    "and prediction."
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.markdown(
        '<div class="main-title">'
        'Stroke Risk Intelligence Dashboard'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Interactive analysis of stroke-related risk factors and '
        'machine-learning based risk prediction.'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # KPI CALCULATIONS
    # --------------------------------------------------------

    total_records = len(df)

    at_risk = int(
        df["At Risk (Binary)"].sum()
    )

    not_at_risk = (
        total_records - at_risk
    )

    average_risk = (
        df["Stroke Risk (%)"].mean()
    )

    average_age = (
        df["Age"].mean()
    )

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">
                    Total Records
                </div>

                <div class="kpi-value">
                    {total_records:,}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">
                    At Risk
                </div>

                <div class="kpi-value">
                    {at_risk:,}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">
                    Average Risk
                </div>

                <div class="kpi-value">
                    {average_risk:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:

        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">
                    Average Age
                </div>

                <div class="kpi-value">
                    {average_age:.1f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # --------------------------------------------------------
    # RISK DISTRIBUTION
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    # Pie chart

    with col1:

        risk_counts = (
            df["At Risk (Binary)"]
            .value_counts()
            .reset_index()
        )

        risk_counts.columns = [
            "At Risk",
            "Count"
        ]

        risk_counts["Status"] = (
            risk_counts["At Risk"]
            .map({
                0: "Not At Risk",
                1: "At Risk"
            })
        )

        fig = px.pie(
            risk_counts,
            names="Status",
            values="Count",
            title="At-Risk Distribution",
            hole=0.45
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # Age distribution

    with col2:

        age_df = df.copy()

        age_df["Risk Status"] = (
            age_df["At Risk (Binary)"]
            .map({
                0: "Not At Risk",
                1: "At Risk"
            })
        )

        fig = px.histogram(
            age_df,
            x="Age",
            color="Risk Status",
            nbins=30,
            title="Age Distribution by Risk Status",
            labels={
                "Age": "Age",
                "Risk Status": "Risk Status"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # RISK CATEGORY
    # --------------------------------------------------------

    if "Risk Category" in df.columns:

        risk_category_counts = (
            df["Risk Category"]
            .value_counts()
            .reset_index()
        )

        risk_category_counts.columns = [
            "Risk Category",
            "Count"
        ]

        fig = px.bar(
            risk_category_counts,
            x="Risk Category",
            y="Count",
            text="Count",
            title="Stroke Risk Category Distribution"
        )

        fig.update_traces(
            textposition="outside"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # DATASET INFORMATION
    # --------------------------------------------------------

    st.markdown("### Dataset Overview")

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:

        st.metric(
            "At-Risk Patients",
            f"{at_risk:,}"
        )

    with info_col2:

        st.metric(
            "Not At-Risk Patients",
            f"{not_at_risk:,}"
        )

    with info_col3:

        at_risk_percentage = (
            at_risk / total_records * 100
        )

        st.metric(
            "At-Risk Percentage",
            f"{at_risk_percentage:.2f}%"
        )


# ============================================================
# RISK FACTOR ANALYSIS
# ============================================================

elif page == "Risk Factor Analysis":

    st.markdown(
        '<div class="main-title">'
        'Risk Factor Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Explore the distribution of individual stroke-related '
        'risk factors in the dataset.'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # RISK FACTOR LIST
    # --------------------------------------------------------

    risk_factor_columns = [

        "Chest Pain",

        "Shortness of Breath",

        "Irregular Heartbeat",

        "Fatigue & Weakness",

        "Dizziness",

        "Swelling (Edema)",

        "Pain in Neck/Jaw/Shoulder/Back",

        "Excessive Sweating",

        "Persistent Cough",

        "Nausea/Vomiting",

        "High Blood Pressure",

        "Chest Discomfort (Activity)",

        "Cold Hands/Feet",

        "Snoring/Sleep Apnea",

        "Anxiety/Feeling of Doom"

    ]

    available_factors = [

        factor

        for factor in risk_factor_columns

        if factor in df.columns

    ]

    # --------------------------------------------------------
    # SELECT FACTOR
    # --------------------------------------------------------

    selected_factor = st.selectbox(
        "Select a Risk Factor",
        available_factors
    )

    # --------------------------------------------------------
    # SELECTED FACTOR DISTRIBUTION
    # --------------------------------------------------------

    factor_counts = (
        df[selected_factor]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    factor_counts.columns = [
        "Value",
        "Count"
    ]

    factor_counts["Status"] = (
        factor_counts["Value"]
        .map({
            0: "Absent",
            1: "Present"
        })
    )

    fig = px.bar(
        factor_counts,
        x="Status",
        y="Count",
        text="Count",
        title=f"{selected_factor} Distribution"
    )

    fig.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # RISK RATE BY FACTOR
    # --------------------------------------------------------

    factor_summary = []

    for factor in available_factors:

        present_records = (
            df[df[factor] == 1]
        )

        if len(present_records) > 0:

            risk_rate = (
                present_records[
                    "At Risk (Binary)"
                ].mean() * 100
            )

            factor_summary.append({

                "Risk Factor": factor,

                "At-Risk Percentage": risk_rate

            })

    factor_summary = pd.DataFrame(
        factor_summary
    )

    factor_summary = (
        factor_summary
        .sort_values(
            "At-Risk Percentage",
            ascending=False
        )
    )

    fig = px.bar(
        factor_summary,
        x="At-Risk Percentage",
        y="Risk Factor",
        orientation="h",
        title=(
            "At-Risk Percentage When "
            "Risk Factor Is Present"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # NOTE
    # --------------------------------------------------------

    st.markdown(
        '<div class="info-box">'
        '<b>Interpretation Note:</b> '
        'These visualizations describe patterns within the supplied '
        'dataset. They should not be interpreted as clinical evidence '
        'of causation.'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# PATIENT RISK PROFILE
# ============================================================

elif page == "Patient Risk Profile":

    st.markdown(
        '<div class="main-title">'
        'Patient Risk Profile'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Filter the dataset and inspect patient-level risk patterns.'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        minimum_age = int(
            df["Age"].min()
        )

        maximum_age = int(
            df["Age"].max()
        )

        age_range = st.slider(
            "Age Range",
            min_value=minimum_age,
            max_value=maximum_age,
            value=(
                minimum_age,
                maximum_age
            )
        )

    with col2:

        risk_status = st.selectbox(
            "Risk Status",
            [
                "All",
                "At Risk",
                "Not At Risk"
            ]
        )

    # --------------------------------------------------------
    # FILTER DATA
    # --------------------------------------------------------

    filtered_df = df[
        (df["Age"] >= age_range[0]) &
        (df["Age"] <= age_range[1])
    ].copy()

    if risk_status == "At Risk":

        filtered_df = filtered_df[
            filtered_df["At Risk (Binary)"] == 1
        ]

    elif risk_status == "Not At Risk":

        filtered_df = filtered_df[
            filtered_df["At Risk (Binary)"] == 0
        ]

    st.write(
        f"Showing **{len(filtered_df):,}** records"
    )

    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    display_columns = [

        "Age",

        "Stroke Risk (%)",

        "At Risk (Binary)"

    ]

    if "Risk Category" in filtered_df.columns:

        display_columns.append(
            "Risk Category"
        )

    display_columns = [

        column

        for column in display_columns

        if column in filtered_df.columns

    ]

    st.dataframe(
        filtered_df[
            display_columns
        ].head(500),
        use_container_width=True
    )

    # --------------------------------------------------------
    # FILTER SUMMARY
    # --------------------------------------------------------

    if len(filtered_df) > 0:

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Average Age",
                f"{filtered_df['Age'].mean():.1f}"
            )

        with col2:

            st.metric(
                "Average Stroke Risk",
                f"{filtered_df['Stroke Risk (%)'].mean():.2f}%"
            )

        with col3:

            st.metric(
                "At-Risk Rate",
                f"{filtered_df['At Risk (Binary)'].mean() * 100:.2f}%"
            )


# ============================================================
# STROKE RISK PREDICTION
# ============================================================

elif page == "Stroke Risk Prediction":

    st.markdown(
        '<div class="main-title">'
        'Stroke Risk Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Enter patient risk-factor information to obtain a '
        'machine-learning model prediction.'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    st.markdown(
        '<div class="warning-box">'
        '<b>Academic Use Only:</b> '
        'This prediction is generated by a machine-learning model '
        'trained on the supplied dataset. It is not a medical diagnosis '
        'and must not be used for clinical decision-making.'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # AGE
    # --------------------------------------------------------

    st.markdown("### Patient Information")

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=90,
        value=50,
        step=1
    )

    # --------------------------------------------------------
    # RISK FACTORS
    # --------------------------------------------------------

    st.markdown("### Risk Factors")

    input_values = {}

    # Use only the features expected by the trained model.
    # Age is handled separately.

    non_age_features = [

        feature

        for feature in feature_columns

        if feature != "Age"

    ]

    left_column, right_column = st.columns(2)

    for index, feature in enumerate(
        non_age_features
    ):

        target_column = (
            left_column
            if index % 2 == 0
            else right_column
        )

        with target_column:

            input_values[feature] = st.selectbox(

                feature,

                options=[0, 1],

                format_func=lambda value:
                    "Present"
                    if value == 1
                    else "Absent",

                key=f"prediction_{feature}"

            )

    # --------------------------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------------------------

    if st.button(
        "Predict Stroke Risk",
        type="primary",
        use_container_width=True
    ):

        # Build input dictionary

        input_data = {

            "Age": age

        }

        input_data.update(
            input_values
        )

        # Create DataFrame in EXACT
        # training feature order

        input_df = pd.DataFrame(
            [input_data],
            columns=feature_columns
        )

        # Model prediction

        prediction = model.predict(
            input_df
        )[0]

        # Probability

        if hasattr(
            model,
            "predict_proba"
        ):

            probability = (
                model.predict_proba(
                    input_df
                )[0][1]
            )

        else:

            probability = float(prediction)

        st.markdown("---")

        # ----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------

        if prediction == 1:

            st.error(
                "MODEL PREDICTION: AT RISK"
            )

        else:

            st.success(
                "MODEL PREDICTION: NOT AT RISK"
            )

        st.metric(
            "Predicted Risk Probability",
            f"{probability * 100:.2f}%"
        )

        # ----------------------------------------------------
        # PROBABILITY BAR
        # ----------------------------------------------------

        st.progress(
            float(probability)
        )

        # ----------------------------------------------------
        # RESULT INTERPRETATION
        # ----------------------------------------------------

        if prediction == 1:

            st.warning(
                "The trained model classified the entered "
                "record as At Risk."
            )

        else:

            st.info(
                "The trained model classified the entered "
                "record as Not At Risk."
            )

        st.markdown(
            '<div class="info-box">'
            '<b>Important:</b> '
            'The displayed probability is the output of the trained '
            'machine-learning model for the supplied input values. '
            'It is not a clinically validated probability.'
            '</div>',
            unsafe_allow_html=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Stroke Risk Analysis and Prediction Using Machine Learning | "
    "Academic Data Science Project"
)
