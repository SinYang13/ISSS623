"""Streamlit webapp: colorectal cancer screening non-compliance risk predictor.

Project: Predicting colorectal cancer screening non-compliance to guide targeted outreach.
Uses the project's official model (`gb_M3_tuned_calibrated`, gradient boosting, tuned,
isotonic-calibrated in `5 Evaluation.ipynb`) via the same `predict_respondents()` function used in
`9 Predict.ipynb` -- both import from `pipeline_components.py`, so the notebook and this app can
never drift out of sync with each other.

Run with:
    pip install streamlit
    streamlit run app.py
(run from inside the `code/` folder, so the relative ../data path resolves correctly)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from pipeline_components import (
    M1_FEATURES, M2_FEATURES, M3_FEATURES, ORDINAL_FEATURES,
    load_prediction_context, predict_respondents,
)

DATA_DIR = Path(__file__).parent / ".." / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROCESSED_DIR / "modelling"
ANALYTIC_PATH = PROCESSED_DIR / "crc_analytic_dataset.csv"
MODEL_NAME = "gb_M3_tuned_calibrated"

FEATURE_LABELS = {
    "age_group": "Age group", "sex": "Sex", "race_ethnicity": "Race / ethnicity",
    "education_level": "Education level", "income_group": "Household income",
    "employment_status": "Employment status", "marital_status": "Marital status",
    "urban_rural": "Urban / rural", "insurance_status": "Insurance status",
    "personal_doctor": "Has a personal doctor", "cost_barrier": "Cost was a barrier to care",
    "checkup_recency": "Time since last general check-up", "general_health": "Self-rated general health",
    "diabetes_status": "Diabetes status", "heart_disease": "Heart disease / MI history",
    "smoking_status": "Smoking status", "bmi_category": "BMI category",
}

# A sensible default value per feature (the most common category in the training data, chosen so
# the form starts on a plausible respondent rather than an arbitrary alphabetical first option).
DEFAULT_VALUES = {
    "age_group": "60-64", "sex": "Female", "race_ethnicity": "White only, non-Hispanic",
    "education_level": "Attended college or technical school", "income_group": "$50,000 to < $100,000",
    "employment_status": "Employed for wages", "marital_status": "Married", "urban_rural": "Urban",
    "insurance_status": "Insured", "personal_doctor": "Has personal doctor", "cost_barrier": "No",
    "checkup_recency": "Within past year", "general_health": "Good", "diabetes_status": "No diabetes",
    "heart_disease": "CHD or MI not reported", "smoking_status": "Never smoked", "bmi_category": "Overweight",
}

ORDINAL_ORDER_HINT = {
    "age_group": ["45-49", "50-54", "55-59", "60-64", "65-69", "70-74",
                  "75 (source category 75-79)", "Not reported"],
    "income_group": ["Less than $15,000", "$15,000 to < $25,000", "$25,000 to < $35,000",
                      "$35,000 to < $50,000", "$50,000 to < $100,000", "$100,000 to < $200,000",
                      "$200,000 or more", "Not reported"],
    "education_level": ["Did not graduate high school", "Graduated high school",
                         "Attended college or technical school",
                         "Graduated college or technical school", "Not reported"],
    "general_health": ["Poor", "Fair", "Good", "Very good", "Excellent", "Not reported"],
}


@st.cache_resource
def get_prediction_context():
    return load_prediction_context(MODEL_DIR, ANALYTIC_PATH, MODEL_NAME)


st.set_page_config(page_title="CRC Screening Non-Compliance Risk", layout="centered")
st.title("Colorectal Cancer Screening Non-Compliance Risk")
st.caption(
    "Predicting colorectal cancer screening non-compliance to guide targeted outreach -- "
    f"uses the project's official model (`{MODEL_NAME}`), calibrated so predicted risk reflects "
    "the true observed rate, not just a ranking score."
)

with st.spinner("Loading model..."):
    model, reference_risk_scores, valid_categories = get_prediction_context()

st.markdown("### Respondent details")

domains = [
    ("Demographic and socioeconomic (M1)", M1_FEATURES),
    ("Healthcare access (M2)", [f for f in M2_FEATURES if f not in M1_FEATURES]),
    ("Health status (M3)", [f for f in M3_FEATURES if f not in M2_FEATURES]),
]

responses = {}
for domain_title, domain_features in domains:
    st.markdown(f"**{domain_title}**")
    cols = st.columns(2)
    for i, feature in enumerate(domain_features):
        options = ORDINAL_ORDER_HINT.get(feature, valid_categories[feature])
        # valid_categories is the source of truth; ORDINAL_ORDER_HINT is only used to order the
        # dropdown sensibly for the 3 ordinal features that have one -- fall back safely if the
        # data's actual categories ever diverge from the hint.
        options = [o for o in options if o in valid_categories[feature]] or valid_categories[feature]
        default_idx = options.index(DEFAULT_VALUES[feature]) if DEFAULT_VALUES[feature] in options else 0
        with cols[i % 2]:
            responses[feature] = st.selectbox(
                FEATURE_LABELS[feature], options, index=default_idx, key=feature,
            )

st.markdown("---")

if st.button("Predict risk", type="primary"):
    try:
        result = predict_respondents(model, valid_categories, reference_risk_scores, [responses])
        risk = float(result.loc[0, "predicted_risk"])
        percentile = float(result.loc[0, "percentile_vs_test_set"])
        decile = result.loc[0, "approx_risk_decile"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted non-compliance risk", f"{risk:.1%}")
        col2.metric("Percentile vs. test population", f"{percentile:.0f}th")
        col3.metric("Approximate risk decile", decile)

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.hist(reference_risk_scores, bins=40, color="#B0B8C4", label="Test-set risk distribution")
        ax.axvline(risk, color="#C44E52", linewidth=2.5, label=f"This respondent ({risk:.1%})")
        ax.set_xlabel("Predicted non-compliance risk")
        ax.set_ylabel("Number of test respondents")
        ax.legend(loc="upper right", fontsize=9)
        st.pyplot(fig)

        if percentile >= 90:
            st.info(
                "This respondent falls in the top decile of predicted risk -- the group the "
                "project's risk-decile analysis (`5 Evaluation.ipynb`) found captures roughly "
                "28% of all non-compliant adults at 10% outreach capacity."
            )
    except ValueError as e:
        st.error(str(e))

st.markdown("---")
st.caption(
    "Model: gradient boosting (tuned), isotonic-calibrated on a held-out calibration split. "
    "Trained and evaluated on 2024 BRFSS data for US adults aged 45-75. Predictions are for "
    "prediction/targeting purposes, not a clinical diagnosis or a causal risk estimate -- see the "
    "project proposal's Risks and Mitigations section for caveats (self-reported screening status, "
    "US-to-Singapore transferability, unweighted analysis)."
)
