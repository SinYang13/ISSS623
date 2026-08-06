"""Shared preprocessing components for the BRFSS colorectal-screening modelling notebooks.

Defined once here (not copy-pasted per notebook) specifically so fitted pipelines that use
`OrdinalRankWithMissingFlag` can be pickled with `joblib` and reloaded in a different notebook or
session -- a class defined inline in a notebook cell has no stable, importable module path, so
joblib/pickle cannot serialise it. Every notebook that builds or loads a pipeline should import
from this module rather than redefining these pieces locally.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

RANDOM_SEED = 42

demographic_features = ["age_group", "sex", "race_ethnicity", "education_level", "income_group",
                        "employment_status", "marital_status", "urban_rural"]
access_features = ["insurance_status", "personal_doctor", "cost_barrier", "checkup_recency"]
health_features = ["general_health", "diabetes_status", "heart_disease", "smoking_status", "bmi_category"]

M1_FEATURES = demographic_features
M2_FEATURES = M1_FEATURES + access_features
M3_FEATURES = M2_FEATURES + health_features
assert len(M1_FEATURES) == 8 and len(M2_FEATURES) == 12 and len(M3_FEATURES) == 17

ORDINAL_FEATURES = ["age_group", "income_group", "education_level", "general_health"]
NOMINAL_FEATURES = [f for f in M3_FEATURES if f not in ORDINAL_FEATURES]

ORDINAL_CATEGORY_ORDER = {
    "age_group": ["45-49", "50-54", "55-59", "60-64", "65-69", "70-74",
                  "75 (source category 75-79)"],
    "income_group": ["Less than $15,000", "$15,000 to < $25,000", "$25,000 to < $35,000",
                      "$35,000 to < $50,000", "$50,000 to < $100,000", "$100,000 to < $200,000",
                      "$200,000 or more"],
    "education_level": ["Did not graduate high school", "Graduated high school",
                         "Attended college or technical school",
                         "Graduated college or technical school"],
    "general_health": ["Poor", "Fair", "Good", "Very good", "Excellent"],
}


class OrdinalRankWithMissingFlag(BaseEstimator, TransformerMixin):
    """Maps each ordinal column to its rank (0..k-1) using an explicit, pre-defined category
    order, plus a companion 0/1 'not reported' flag, so the missing category never sits inside
    the ordinal scale as if it were a value. The category order is supplied at construction time
    (derived from the codebook, not fit from data), so `fit` only needs to record the column
    order it was given."""

    def __init__(self, category_orders):
        self.category_orders = category_orders

    def fit(self, X, y=None):
        self.columns_ = list(X.columns) if hasattr(X, "columns") else list(self.category_orders)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_)
        out = {}
        for col in self.columns_:
            rank_map = {cat: rank for rank, cat in enumerate(self.category_orders[col])}
            ranks = X[col].map(rank_map)
            out[f"{col}__rank"] = ranks.fillna(-1).astype(float)
            out[f"{col}__not_reported"] = ranks.isna().astype(float)
        return pd.DataFrame(out, index=X.index).values

    def get_feature_names_out(self, input_features=None):
        names = []
        for col in self.columns_:
            names += [f"{col}__rank", f"{col}__not_reported"]
        return np.array(names)


def ordinal_subset(features):
    return [f for f in features if f in ORDINAL_FEATURES]


def nominal_subset(features):
    return [f for f in features if f in NOMINAL_FEATURES]


def build_logreg_pipeline(features):
    pre = ColumnTransformer([
        ("ordinal", OrdinalRankWithMissingFlag(
            {c: ORDINAL_CATEGORY_ORDER[c] for c in ordinal_subset(features)}
        ), ordinal_subset(features)),
        ("nominal", OneHotEncoder(handle_unknown="ignore"), nominal_subset(features)),
    ])
    return Pipeline([
        ("preprocess", pre),
        ("scale", StandardScaler(with_mean=False)),
        ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
    ])


def build_gb_pipeline(features):
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), features),
    ])
    return Pipeline([
        ("preprocess", pre),
        ("model", HistGradientBoostingClassifier(
            categorical_features=[True] * len(features), random_state=RANDOM_SEED,
        )),
    ])
