"""
Task 1: Adversity Profile

A trade is "adverse" at horizon tau (tau seconds after execution) if closing the
position at that horizon loses money for the LP:

    PnL(t=tau) = side * V * (M_tau - T_P) < 0

Volume V is always positive, so the sign only depends on side*(M_tau - T_P). A
client's adversity at tau is just the % of its trades that are adverse at tau.

Submission function: adversity_profile(client, tau) -> List[float]
"""

import os
from typing import List

import pandas as pd


# Data loading, cached so repeated calls stay cheap.
_DATA_CACHE = {}


def _data_path() -> str:
    """Locate trade_data.csv next to this file or in the parent directory."""
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (
        os.path.join(here, "trade_data.csv"),
        os.path.join(os.path.dirname(here), "trade_data.csv"),
        "trade_data.csv",
    ):
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("trade_data.csv not found.")


def _load() -> pd.DataFrame:
    if "df" not in _DATA_CACHE:
        df = pd.read_csv(_data_path())
        _DATA_CACHE["df"] = df
    return _DATA_CACHE["df"]


# horizon (seconds) -> mid column name
_MID_COL = {5: "M5", 10: "M10", 15: "M15", 20: "M20", 25: "M25", 30: "M30"}


def adversity_profile(client: str, tau: List[int]) -> List[float]:
    """
    Parameters
    ----------
    client : str
        Client identifier (single character or string), e.g. "A".
    tau : List[int]
        List of horizons in seconds, e.g. [5, 10, 15, 20, 25, 30].

    Returns
    -------
    List[float]
        Adversity percentage (0-100) at each horizon, in the same order as
        ``tau``.
    """
    df = _load()
    sub = df[df["Name"] == client]

    side = sub["Side"].to_numpy()
    tp = sub["Trade Price"].to_numpy()

    out: List[float] = []
    n = len(sub)
    for t in tau:
        col = _MID_COL.get(int(t))
        if col is None:
            raise ValueError(f"Unsupported horizon tau={t}; expected one of {sorted(_MID_COL)}")
        if n == 0:
            out.append(float("nan"))
            continue
        m_tau = sub[col].to_numpy()
        pnl = side * (m_tau - tp)          # V > 0 omitted: it does not change the sign
        adverse_pct = 100.0 * (pnl < 0).mean()
        out.append(float(adverse_pct))
    return out


# Write out task1_results.csv for all clients.
def _build_results_csv(path: str = "task1_results.csv") -> pd.DataFrame:
    tau = [5, 10, 15, 20, 25, 30]
    df = _load()
    clients = sorted(df["Name"].unique())

    rows = []
    for c in clients:
        prof = adversity_profile(c, tau)
        rows.append([c] + prof)

    cols = ["client"] + [f"tau={t}" for t in tau]
    res = pd.DataFrame(rows, columns=cols)
    res.to_csv(path, index=False)
    return res


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    res = _build_results_csv(os.path.join(out_dir, "task1_results.csv"))
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(res.to_string(index=False))
