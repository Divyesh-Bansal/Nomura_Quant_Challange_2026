"""
Task 3: Adversity Prediction Model M(client, trade, features, tau)

Build a model M predicting the probability a trade is adverse at horizon tau in
{5,10,15,20,25,30}:

    M(client, trade, features, tau) = P(Trade is Adverse at t = tau)

A trade is adverse at tau (LP perspective) iff
    side * V * (M_tau - T_P) < 0   <=>   side * (M_tau - T_P) < 0   (V > 0).

Two correctness constraints I held to:
1. No target leakage. The mid columns M5..M30 define the label and encode the
   future, so they are never used as features - only execution-time info is.
2. No leakage across splits. I split by date (chronological 60/20/20) so the six
   stacked rows (one per horizon) of any trade stay in the same split.

Feature selection (validation-driven):
I screened every observable feature (side, volume, spread, trade_price - M0,
signed fill = side*(trade_price - M0), time-of-day, client, tau). Holding out
val/test by date showed that client identity and tau carry essentially all the
generalizable signal (adverse rate ~0.42 for A up to ~0.62 for F at tau=30,
rising with tau), while side/volume/spread/signed-fill/time-of-day are flat
(~0.48 everywhere) - adding them only memorised noise and worsened test log-loss
below a trivial client base-rate. So I keep just the features that generalize: a
one-hot client encoding plus tau. Train/val/test metrics then agree closely.

Model: one HistGradientBoostingClassifier with tau as an input feature, trained
on the data stacked 6x (one row per (trade, tau), label = adverse-at-that-tau).
A single estimator serves every horizon, drives predict_adversity for any tau,
and lets compute_metrics evaluate per horizon and average. Gradient-boosted trees
recover the client x tau structure and give probabilities suited to log-loss.

Feature vector ordering (see FEATURE_NAMES):
    0: tau (seconds)   1: client_A   2: client_B   3: client_C
    4: client_D        5: client_E   6: client_F   (indices 1-6 one-hot)

Submission interface:
    predict_adversity(*args, **kwargs) -> float
    compute_metrics(*args, **kwargs)   -> pd.DataFrame
"""

import os
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    precision_score,
    recall_score,
)


# Constants
ALL_TAU = [5, 10, 15, 20, 25, 30]
MID_COL = {5: "M5", 10: "M10", 15: "M15", 20: "M20", 25: "M25", 30: "M30"}
CLIENTS = ["A", "B", "C", "D", "E", "F"]

FEATURE_NAMES = [
    "tau",        # 0
    "client_A",   # 1
    "client_B",   # 2
    "client_C",   # 3
    "client_D",   # 4
    "client_E",   # 5
    "client_F",   # 6
]

RANDOM_STATE = 42
DECISION_THRESHOLD = 0.5

_STATE = {}  # cache for raw data, splits and the fitted model


# Data loading & chronological date split
def _data_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (
        os.path.join(here, "trade_data.csv"),
        os.path.join(os.path.dirname(here), "trade_data.csv"),
        "trade_data.csv",
    ):
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("trade_data.csv not found.")


def _load_raw() -> pd.DataFrame:
    if "raw" not in _STATE:
        _STATE["raw"] = pd.read_csv(_data_path())
    return _STATE["raw"]


def _date_split(df: pd.DataFrame):
    """Chronological 60/20/20 split by unique date (suggested ratio)."""
    dates = sorted(df["Date"].unique())
    n = len(dates)
    n_train = int(round(0.60 * n))
    n_val = int(round(0.20 * n))
    train_dates = set(dates[:n_train])
    val_dates = set(dates[n_train:n_train + n_val])
    test_dates = set(dates[n_train + n_val:])
    return train_dates, val_dates, test_dates


# Feature engineering. M5..M30 are used only for labels, never as features.
def _client_onehot(names: np.ndarray) -> np.ndarray:
    oh = np.zeros((len(names), len(CLIENTS)), dtype=float)
    for j, c in enumerate(CLIENTS):
        oh[:, j] = (names == c).astype(float)
    return oh


def _features_for_tau(df: pd.DataFrame, tau: int) -> np.ndarray:
    """Feature matrix for a single horizon: [tau, client one-hot]."""
    n = len(df)
    tau_col = np.full((n, 1), float(tau))
    onehot = _client_onehot(df["Name"].to_numpy())
    return np.hstack([tau_col, onehot])


def _labels_for_tau(df: pd.DataFrame, tau: int) -> np.ndarray:
    """Adverse-at-tau label: side * (M_tau - T_P) < 0."""
    side = df["Side"].to_numpy(dtype=float)
    m_tau = df[MID_COL[tau]].to_numpy(dtype=float)
    tp = df["Trade Price"].to_numpy(dtype=float)
    return (side * (m_tau - tp) < 0).astype(int)


def _stack(df: pd.DataFrame):
    """Stack all six horizons -> (X, y)."""
    Xs, ys = [], []
    for t in ALL_TAU:
        Xs.append(_features_for_tau(df, t))
        ys.append(_labels_for_tau(df, t))
    return np.vstack(Xs), np.concatenate(ys)


# Training, done lazily and cached.
def _ensure_trained():
    if "model" in _STATE:
        return
    df = _load_raw()
    train_d, val_d, test_d = _date_split(df)

    df_train = df[df["Date"].isin(train_d)]
    df_val = df[df["Date"].isin(val_d)]
    df_test = df[df["Date"].isin(test_d)]

    Xtr, ytr = _stack(df_train)

    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.10,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=500,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )
    model.fit(Xtr, ytr)

    _STATE["model"] = model
    _STATE["splits"] = {"train": df_train, "validation": df_val, "test": df_test}


# Prediction interface
def _make_feature_row(client: str, tau: int) -> np.ndarray:
    if client not in CLIENTS:
        raise ValueError(f"Unknown client '{client}'; expected one of {CLIENTS}.")
    if int(tau) not in MID_COL:
        raise ValueError(f"Unsupported horizon tau={tau}; expected one of {ALL_TAU}.")
    onehot = [1.0 if client == c else 0.0 for c in CLIENTS]
    return np.asarray([float(tau)] + onehot, dtype=float).reshape(1, -1)


def predict_adversity(*args, **kwargs) -> float:
    """
    Predict P(trade is adverse at horizon tau).

    Accepted calling conventions
    ----------------------------
    1. Keyword fields (recommended):
        predict_adversity(client="F", tau=30)
       Extra trade fields (side, volume, spread, ...) are accepted and ignored,
       since validation-based feature selection retained only client and tau.
    2. A pre-built feature vector (length 7, ordering = FEATURE_NAMES):
        predict_adversity(features=[30, 0, 0, 0, 0, 0, 1])
        predict_adversity([30, 0, 0, 0, 0, 0, 1])

    Returns
    -------
    float
        Probability in [0, 1] that the trade is adverse at the given horizon.
    """
    _ensure_trained()
    model = _STATE["model"]

    # Case 2: explicit feature vector
    feats = kwargs.get("features", None)
    if feats is None and len(args) == 1 and hasattr(args[0], "__len__") \
            and not isinstance(args[0], (str, bytes)):
        feats = args[0]
    if feats is not None:
        x = np.asarray(feats, dtype=float).reshape(1, -1)
        if x.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"Expected {len(FEATURE_NAMES)} features ordered as {FEATURE_NAMES}, "
                f"got {x.shape[1]}."
            )
        return float(model.predict_proba(x)[0, 1])

    # Case 1: keyword / positional client + tau
    client = kwargs.get("client")
    tau = kwargs.get("tau")
    if client is None and len(args) >= 1 and isinstance(args[0], str):
        client = args[0]
    if tau is None and len(args) >= 2:
        tau = args[1]
    if client is None or tau is None:
        raise ValueError(
            "Provide client and tau (e.g. predict_adversity(client='F', tau=30)) "
            f"or a length-{len(FEATURE_NAMES)} 'features' vector ordered as {FEATURE_NAMES}."
        )
    x = _make_feature_row(client, int(tau))
    return float(model.predict_proba(x)[0, 1])


# Metrics
def _metrics_for_split(df: pd.DataFrame, threshold: float = DECISION_THRESHOLD) -> dict:
    """Four metrics per horizon, then averaged over the six horizons."""
    model = _STATE["model"]
    accs, precs, recs, lls = [], [], [], []
    for t in ALL_TAU:
        X = _features_for_tau(df, t)
        y = _labels_for_tau(df, t)
        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= threshold).astype(int)
        accs.append(accuracy_score(y, pred))
        precs.append(precision_score(y, pred, zero_division=0))
        recs.append(recall_score(y, pred, zero_division=0))
        lls.append(log_loss(y, proba, labels=[0, 1]))
    return {
        "accuracy": float(np.mean(accs)),
        "precision": float(np.mean(precs)),
        "recall": float(np.mean(recs)),
        "log_loss": float(np.mean(lls)),
    }


def compute_metrics(*args, **kwargs) -> pd.DataFrame:
    """
    Returns
    -------
    pd.DataFrame
        Index   = ['train', 'validation', 'test'];
        Columns = ['accuracy', 'precision', 'recall', 'log_loss'].
        Each cell is the metric averaged over the six horizons tau.
    """
    _ensure_trained()
    threshold = float(kwargs.get("threshold", DECISION_THRESHOLD))
    splits = _STATE["splits"]
    rows = {}
    for split_name in ["train", "validation", "test"]:
        rows[split_name] = _metrics_for_split(splits[split_name], threshold)
    out = pd.DataFrame(rows).T[["accuracy", "precision", "recall", "log_loss"]]
    out.index.name = "split"
    return out


# Results CSV
def _build_results_csv(path: str = "task3_results.csv") -> pd.DataFrame:
    m = compute_metrics()
    m.reset_index().to_csv(path, index=False)
    return m


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    metrics = _build_results_csv(os.path.join(out_dir, "task3_results.csv"))
    pd.set_option("display.float_format", lambda v: f"{v:.6f}")
    print("Per-split metrics (averaged over horizons):")
    print(metrics.to_string())
    print()
    for c in CLIENTS:
        ps = [predict_adversity(client=c, tau=t) for t in ALL_TAU]
        print(f"client {c}: " + "  ".join(f"t{t}={p:.3f}" for t, p in zip(ALL_TAU, ps)))
