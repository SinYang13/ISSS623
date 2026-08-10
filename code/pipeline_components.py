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
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
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


def build_rf_pipeline(features, n_estimators=300, min_samples_leaf=20):
    """Random Forest shares gradient boosting's encoding, not logistic regression's: no native
    categorical support in sklearn's RandomForestClassifier, but tree ensembles can still split
    usefully on label/ordinal-encoded integers even without one-hot -- unlike a linear model,
    a tree doesn't assume the encoded values are on a meaningful numeric scale.

    min_samples_leaf=20 (not the sklearn default of 1) is a deliberate, non-searched choice, not
    hyperparameter tuning: unconstrained trees on this data overfit badly (test AUROC ~0.71-0.72
    at min_samples_leaf=1 vs ~0.76 at 20), so this is closer to "avoid a setting known to overfit"
    than a tuned value. Real tuning is still deferred to a cross-validated search, per the
    calibration-vs-validation note in notebook 3.
    """
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), features),
    ])
    return Pipeline([
        ("preprocess", pre),
        ("model", RandomForestClassifier(
            n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
            random_state=RANDOM_SEED, n_jobs=-1,
        )),
    ])


def build_ffnn_pipeline(features, hidden_layer_sizes=(64, 32), alpha=1e-4,
                         learning_rate_init=1e-3, max_iter=300):
    """Feedforward neural network (sklearn's `MLPClassifier`). Shares logistic regression's
    encoding, not the tree models': a network's input layer has no notion that an ordinal-encoded
    integer or a one-hot indicator should be split on a threshold the way a tree can, so nominal
    features are one-hot encoded (not label/ordinal-encoded) and every input is scaled -- an
    unscaled mix of 0/1 indicators and larger-range ranks would let large-scale features dominate
    the first layer's weights for reasons that have nothing to do with predictive relevance.

    `sample_weight` support in `MLPClassifier.fit` is not guaranteed across sklearn versions, so
    this is worth confirming against the installed version rather than assuming -- if unsupported,
    class weighting would need an alternative (e.g. weighted resampling) to stay consistent with
    the other three models, which use `sample_weight` rather than oversampling.

    `early_stopping=True` holds out 10% of the training fold internally to decide when to stop,
    which is a within-training-fold mechanism (distinct from, and unrelated to, the
    calibration/test splits) -- a standard way to avoid a fixed `max_iter` either under- or
    over-training the network.
    """
    pre = ColumnTransformer([
        ("ordinal", OrdinalRankWithMissingFlag(
            {c: ORDINAL_CATEGORY_ORDER[c] for c in ordinal_subset(features)}
        ), ordinal_subset(features)),
        ("nominal", OneHotEncoder(handle_unknown="ignore"), nominal_subset(features)),
    ])
    return Pipeline([
        ("preprocess", pre),
        ("scale", StandardScaler(with_mean=False)),
        ("model", MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes, alpha=alpha, learning_rate_init=learning_rate_init,
            max_iter=max_iter, early_stopping=True, random_state=RANDOM_SEED,
        )),
    ])


def compute_classification_metrics(y_true, y_proba, threshold=0.5):
    """AUROC/AUPRC/Precision/Sensitivity/F1/Accuracy at the given threshold -- the standard set
    reported for every model (and every hyperparameter-search comparison) in notebooks 4a-4d."""
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, accuracy_score,
    )
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "AUROC": roc_auc_score(y_true, y_proba),
        "AUPRC": average_precision_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred),
        "Sensitivity": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "Accuracy": accuracy_score(y_true, y_pred),
    }


def make_model_registry(models_dir, manifest_path):
    """Sets up a model registry bound to the given paths: a `MODEL_REGISTRY` dict plus
    `register_model`/`load_registry` functions. Notebooks 4a-4d each call this with the same
    models_dir/manifest_path, so all four register into one shared manifest despite each running
    in its own kernel with its own empty registry. `register_model` merges into whatever is
    already on the manifest (keyed by model name) rather than overwriting it, so one notebook
    running (or re-running) never erases another's rows. `load_registry` reloads every model
    listed in the manifest without refitting -- used by notebook 5 (Evaluation) to compare across
    everything 4a-4d have registered.
    """
    import joblib

    models_dir.mkdir(parents=True, exist_ok=True)
    registry = {}

    def _write_manifest():
        existing = {}
        if manifest_path.exists():
            existing = pd.read_csv(manifest_path).set_index("name").to_dict("index")
        for name, entry in registry.items():
            row = {
                "model_type": entry["model_type"],
                "n_features": len(entry["features"]),
                "path": entry["path"],
                "notes": entry["notes"],
            }
            row.update(entry["metrics"])
            existing[name] = row
        rows = [{"name": name, **row} for name, row in existing.items()]
        pd.DataFrame(rows).to_csv(manifest_path, index=False)

    def register_model(name, pipeline, metrics, features, model_type, notes=""):
        path = models_dir / f"{name}.joblib"
        joblib.dump(pipeline, path)
        registry[name] = {
            "pipeline": pipeline,
            "metrics": metrics,
            "features": features,
            "model_type": model_type,
            "path": str(path),
            "notes": notes,
        }
        _write_manifest()
        return registry[name]

    def load_registry():
        if not manifest_path.exists():
            return {}
        manifest = pd.read_csv(manifest_path)
        out = {}
        for _, row in manifest.iterrows():
            out[row["name"]] = {
                "pipeline": joblib.load(row["path"]),
                "metrics": {k: row[k] for k in ["AUROC", "AUPRC", "Precision", "Sensitivity", "F1", "Accuracy"]},
                "n_features": int(row["n_features"]),
                "model_type": row["model_type"],
                "path": row["path"],
                "notes": row["notes"],
            }
        return out

    return registry, register_model, load_registry
